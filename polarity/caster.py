from typing import Type, Dict, Any, Optional, List, Tuple
import polars as pl
from pydantic import BaseModel, ValidationError
from datetime import date, datetime
import re
import math

class PolarsCaster:
    def __init__(self, schema_mapping: Dict[str, Type[BaseModel]]):
        self.schema_mapping = schema_mapping

        # Date pattern regex-format pairs
        self.date_patterns = [
            # ISO formats
            (r'^\d{4}-\d{2}-\d{2}$', '%Y-%m-%d'),
            (r'^\d{4}/\d{2}/\d{2}$', '%Y/%m/%d'),
            # US formats
            (r'^\d{1,2}/\d{1,2}/\d{4}$', '%m/%d/%Y'),
            (r'^\d{1,2}-\d{1,2}-\d{4}$', '%m-%d-%Y'),
            (r'^\d{1,2}/\d{1,2}/\d{2}$', '%m/%d/%y'),
            (r'^\d{1,2}-\d{1,2}-\d{2}$', '%m-%d-%y'),
            # European formats
            (r'^\d{1,2}/\d{1,2}/\d{4}$', '%d/%m/%Y'),
            (r'^\d{1,2}-\d{1,2}-\d{4}$', '%d-%m-%Y'),
            (r'^\d{1,2}/\d{1,2}/\d{2}$', '%d/%m/%y'),
            (r'^\d{1,2}-\d{1,2}-\d{2}$', '%d-%m-%y'),
        ]

        # Datetime pattern regex-format pairs
        self.datetime_patterns = [
            # ISO formats with timezone
            (r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$', '%Y-%m-%dT%H:%M:%SZ'),
            (r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}[-+]\d{2}:\d{2}$', '%Y-%m-%dT%H:%M:%S%z'),
            (r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}[-+]\d{4}$', None),  # Special handling
            # Standard formats
            (r'^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$', '%Y-%m-%d %H:%M:%S'),
            (r'^\d{4}/\d{2}/\d{2} \d{2}:\d{2}:\d{2}$', '%Y/%m/%d %H:%M:%S'),
            # US/EU formats with time
            (r'^\d{1,2}/\d{1,2}/\d{4} \d{1,2}:\d{2}:\d{2} [AP]M$', '%m/%d/%Y %I:%M:%S %p'),
            (r'^\d{1,2}/\d{1,2}/\d{4} \d{2}:\d{2}:\d{2}$', '%m/%d/%Y %H:%M:%S'),
            (r'^\d{1,2}/\d{1,2}/\d{4} \d{2}:\d{2}:\d{2}$', '%d/%m/%Y %H:%M:%S'),
            (r'^\d{1,2}-\d{1,2}-\d{4} \d{2}:\d{2}:\d{2}$', '%d-%m-%Y %H:%M:%S'),
        ]

    def _get_polars_schema(self, schema: Type[BaseModel]) -> Dict[str, pl.DataType]:
        """Convierte el schema de Pydantic a un schema compatible con Polars."""
        json_schema = schema.model_json_schema()["properties"]
        polars_schema = {}

        for field_name, field_info in json_schema.items():
            field_type = field_info.get("type")
            format_type = field_info.get("format")

            # Handle Optional[] fields (null type with anyOf)
            if field_type == "null" and "anyOf" in field_info:
                for type_info in field_info["anyOf"]:
                    if type_info.get("type") != "null":
                        field_type = type_info.get("type")
                        format_type = type_info.get("format", None)
                        break

            if field_type == "integer":
                polars_schema[field_name] = pl.Int64
            elif field_type == "number":
                polars_schema[field_name] = pl.Float64
            elif field_type == "string":
                if format_type == "date-time":
                    polars_schema[field_name] = pl.Datetime
                elif format_type == "date":
                    polars_schema[field_name] = pl.Date
                else:
                    polars_schema[field_name] = pl.Utf8
            elif field_type == "boolean":
                polars_schema[field_name] = pl.Boolean
            else:
                polars_schema[field_name] = pl.Utf8

        return polars_schema

    def attempt_cast(self, value: Any, target_type: Type) -> Any:
        """Intenta convertir un valor al tipo especificado."""
        if value is None:
            return None

        if isinstance(value, target_type):
            return value

        try:
            if target_type is int:
                if isinstance(value, str):
                    try:
                        float_val = float(value)
                        if float_val.is_integer():
                            return int(float_val)
                        return None
                    except ValueError:
                        return None
                elif isinstance(value, float):
                    if value.is_integer():
                        return int(value)
                    return None
                return int(value)

            elif target_type is float:
                return float(value)

            elif target_type is str:
                return str(value) if value is not None else None

            elif target_type is bool:
                if isinstance(value, str):
                    value_lower = value.lower().strip()
                    if value_lower in ('true', 'yes', '1', 't', 'y'):
                        return True
                    elif value_lower in ('false', 'no', '0', 'f', 'n', ''):
                        return False
                    return bool(value_lower)  # Non-empty strings are True
                elif isinstance(value, (int, float)):
                    if isinstance(value, float) and math.isnan(value):
                        return None
                    return bool(value)
                return bool(value) if value is not None else None

            elif target_type is date:
                return self.parse_date(value)

            elif target_type is datetime:
                return self.parse_datetime(value)

            else:
                return target_type(value)

        except (ValueError, TypeError, OverflowError):
            return None

    def parse_date(self, value: Any) -> Optional[date]:
        """Parse a value into a date object."""
        if value is None:
            return None

        if isinstance(value, date):
            return value

        if isinstance(value, datetime):
            return value.date()

        if isinstance(value, str):
            # Try ISO format first (YYYY-MM-DD)
            try:
                if 'T' in value:
                    date_part = value.split('T')[0]
                    return date.fromisoformat(date_part)
                elif ' ' in value:  # Handle datetime strings
                    date_part = value.split(' ')[0]
                    if '-' in date_part:
                        return date.fromisoformat(date_part)
                    elif '/' in date_part:
                        # Handle date with slashes
                        year, month, day = map(int, date_part.split('/'))
                        return date(year, month, day)
                else:
                    # Try direct ISO format
                    return date.fromisoformat(value)
            except (ValueError, IndexError):
                pass

            # Try different date formats with explicit parsing
            for pattern, format_str in self.date_patterns:
                if re.match(pattern, value):
                    try:
                        dt = datetime.strptime(value, format_str)
                        return dt.date()
                    except ValueError:
                        continue

            # Try to handle common formats manually
            if '/' in value:
                parts = value.split('/')
                if len(parts) == 3:
                    # Try different date orders
                    try:
                        if len(parts[0]) == 4:  # YYYY/MM/DD
                            return date(int(parts[0]), int(parts[1]), int(parts[2]))
                        elif len(parts[2]) == 4:  # MM/DD/YYYY or DD/MM/YYYY
                            # Try both US and EU formats
                            try:
                                return date(int(parts[2]), int(parts[0]), int(parts[1]))
                            except ValueError:
                                try:
                                    return date(int(parts[2]), int(parts[1]), int(parts[0]))
                                except ValueError:
                                    pass
                    except (ValueError, IndexError):
                        pass

            if '-' in value:
                parts = value.split('-')
                if len(parts) == 3:
                    try:
                        if len(parts[0]) == 4:  # YYYY-MM-DD
                            return date(int(parts[0]), int(parts[1]), int(parts[2]))
                        elif len(parts[2]) == 4:  # MM-DD-YYYY or DD-MM-YYYY
                            # Try both US and EU formats
                            try:
                                return date(int(parts[2]), int(parts[0]), int(parts[1]))
                            except ValueError:
                                try:
                                    return date(int(parts[2]), int(parts[1]), int(parts[0]))
                                except ValueError:
                                    pass
                    except (ValueError, IndexError):
                        pass

            # Try to extract date from datetime string
            dt = self.parse_datetime(value)
            if dt:
                return dt.date()

        return None

    def parse_datetime(self, value: Any) -> Optional[datetime]:
        """Parse a value into a datetime object."""
        if value is None:
            return None

        if isinstance(value, datetime):
            return value

        if isinstance(value, date) and not isinstance(value, datetime):
            return datetime.combine(value, datetime.min.time())

        if isinstance(value, str):
            # Handle ISO format with timezones
            try:
                if 'T' in value:
                    if value.endswith('Z'):
                        return datetime.fromisoformat(value.replace('Z', '+00:00'))
                    elif '+' in value or ('-' in value and value.find("T") < value.find("-", value.find("T"))):
                        try:
                            return datetime.fromisoformat(value)
                        except ValueError:
                            # Fix timezone format without colon (e.g., +0500)
                            match = re.search(r'([-+])(\d{2})(\d{2})$', value)
                            if match:
                                sign, hours, minutes = match.groups()
                                fixed_tz = f"{sign}{hours}:{minutes}"
                                value_fixed = value[:-5] + fixed_tz
                                try:
                                    return datetime.fromisoformat(value_fixed)
                                except ValueError:
                                    pass
                    else:
                        return datetime.fromisoformat(value)
                else:
                    # Try parsing non-T format with fromisoformat
                    try:
                        return datetime.fromisoformat(value)
                    except ValueError:
                        pass
            except ValueError:
                pass

            # Try standard datetime patterns
            for pattern, format_str in self.datetime_patterns:
                if re.match(pattern, value):
                    if format_str is None:  # Special handling for formats like +0500
                        match = re.search(r'([-+])(\d{2})(\d{2})$', value)
                        if match:
                            sign, hours, minutes = match.groups()
                            value_mod = value[:-5] + f"{sign}{hours}:{minutes}"
                            try:
                                return datetime.fromisoformat(value_mod)
                            except ValueError:
                                continue
                    else:
                        try:
                            return datetime.strptime(value, format_str)
                        except ValueError:
                            continue

            # Handle date-only strings by adding zero time
            date_val = self.parse_date(value)
            if date_val:
                return datetime.combine(date_val, datetime.min.time())

            # Manual parsing for common formats
            if '/' in value:
                date_part = value.split(' ')[0] if ' ' in value else value
                time_part = value.split(' ')[1] if ' ' in value else "00:00:00"

                parts = date_part.split('/')
                if len(parts) == 3:
                    try:
                        if len(parts[0]) == 4:  # YYYY/MM/DD
                            d = date(int(parts[0]), int(parts[1]), int(parts[2]))
                        elif len(parts[2]) == 4:  # MM/DD/YYYY or DD/MM/YYYY
                            # Try both US and EU formats
                            try:
                                d = date(int(parts[2]), int(parts[0]), int(parts[1]))
                            except ValueError:
                                d = date(int(parts[2]), int(parts[1]), int(parts[0]))

                        # Parse time if present
                        h, m, s = 0, 0, 0
                        if ':' in time_part:
                            time_parts = time_part.split(':')
                            h = int(time_parts[0])
                            m = int(time_parts[1])
                            s = int(time_parts[2]) if len(time_parts) > 2 else 0

                        return datetime(d.year, d.month, d.day, h, m, s)
                    except (ValueError, IndexError):
                        pass

        return None

    def cast_row(self, schema: Type[BaseModel], row: dict) -> dict:
        """Cast a row according to the Pydantic schema."""
        processed_row = row.copy()
        json_schema = schema.model_json_schema()["properties"]

        for field_name, field_info in json_schema.items():
            if field_name not in processed_row:
                continue

            field_value = processed_row[field_name]
            field_type = field_info.get("type")
            format_type = field_info.get("format")

            # Handle Optional fields
            if field_type == "null" and "anyOf" in field_info:
                for type_info in field_info["anyOf"]:
                    if type_info.get("type") != "null":
                        field_type = type_info.get("type")
                        format_type = type_info.get("format")
                        break

            # Convert value based on field type
            if field_type == "integer":
                processed_row[field_name] = self.attempt_cast(field_value, int)
            elif field_type == "number":
                processed_row[field_name] = self.attempt_cast(field_value, float)
            elif field_type == "string":
                if format_type == "date-time":
                    processed_row[field_name] = self.attempt_cast(field_value, datetime)
                elif format_type == "date":
                    processed_row[field_name] = self.attempt_cast(field_value, date)
                else:
                    processed_row[field_name] = self.attempt_cast(field_value, str)
            elif field_type == "boolean":
                processed_row[field_name] = self.attempt_cast(field_value, bool)

        try:
            validated_row = schema(**processed_row)
            return validated_row.model_dump()
        except ValidationError as e:
            raise ValueError(f"Error validando la fila {processed_row}: {e}")

    def cast_dataframe(self, df: pl.DataFrame, schema_name: str) -> pl.DataFrame:
        """Cast a Polars DataFrame according to the specified schema."""
        if schema_name not in self.schema_mapping:
            raise KeyError(f"No existe el schema '{schema_name}' definido.")

        schema = self.schema_mapping[schema_name]

        if df.is_empty():
            polars_schema = self._get_polars_schema(schema)
            return pl.DataFrame(schema=polars_schema)

        df_dict = df.to_dicts()
        casted_rows = []

        for row in df_dict:
            try:
                casted_row = self.cast_row(schema, row)
                casted_rows.append(casted_row)
            except ValueError as e:
                print(e)

        if not casted_rows:
            polars_schema = self._get_polars_schema(schema)
            return pl.DataFrame(schema=polars_schema)

        # Create DataFrame with proper schema inference
        result_df = pl.DataFrame(casted_rows, infer_schema_length=max(100, len(casted_rows)))

        # Apply schema types
        polars_schema = self._get_polars_schema(schema)
        columns_to_cast = {col: dtype for col, dtype in polars_schema.items() if col in result_df.columns}

        if columns_to_cast:
            result_df = result_df.cast(columns_to_cast)

        return result_df
    
    def split_dataframe(
        self,
        df_new: pl.DataFrame,
        df_db: pl.DataFrame,
        pks: List[str],
        schema_name: str
    ) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
        """
        Splits DataFrames into three categories based on primary keys:
        - insert_df: Rows in df_new that don't exist in df_db
        - equals_df: Rows with matching PKs and identical values across all fields
        - update_df: Rows with matching PKs but different values in some fields
        """
        if schema_name not in self.schema_mapping:
            raise KeyError(f"No existe el schema '{schema_name}' definido.")

        schema = self.schema_mapping[schema_name]

        if not pks:
            raise ValueError("Primary key list 'pks' cannot be empty.")

        # Check for primary key columns in original DataFrames BEFORE casting
        for pk in pks:
            if pk not in df_new.columns:
                raise ValueError(f"Primary key '{pk}' does not exist in df_new.")
            if pk not in df_db.columns:
                raise ValueError(f"Primary key '{pk}' does not exist in df_db.")

        # Check for NULL in primary keys BEFORE casting
        for pk in pks:
            null_count_new = df_new.select(pl.col(pk).is_null().sum()).item()
            null_count_db = df_db.select(pl.col(pk).is_null().sum()).item()

            if null_count_new > 0:
                raise ValueError(f"Primary key '{pk}' contains {null_count_new} null values in df_new.")
            if null_count_db > 0:
                raise ValueError(f"Primary key '{pk}' contains {null_count_db} null values in df_db.")

        # Create copies to preserve original data
        original_df_new = df_new.clone()
        original_df_db = df_db.clone()  # noqa: F841

        # Handle empty DataFrames
        if df_new.is_empty():
            polars_schema = self._get_polars_schema(schema)
            empty_df = pl.DataFrame(schema=polars_schema)
            return empty_df, empty_df, empty_df

        if df_db.is_empty():
            empty_df = pl.DataFrame(schema=original_df_new.schema)
            return original_df_new, empty_df, empty_df

        # Cast the DataFrames for comparison purposes only
        casted_new = self.cast_dataframe(df_new, schema_name)
        casted_db = self.cast_dataframe(df_db, schema_name)

        # 1. Find rows for insert (in new but not in db) using original data
        ids_in_db = casted_db.select(pks).to_dict(as_series=False)[pks[0]]
        insert_df = original_df_new.filter(~pl.col(pks[0]).is_in(ids_in_db))

        # For matching IDs, we'll compare the casted values
        ids_in_both = casted_new.filter(pl.col(pks[0]).is_in(ids_in_db)).select(pks).to_dict(as_series=False)[pks[0]]

        equals_ids = []
        update_ids = []

        # Loop through each ID that exists in both dataframes
        for id_val in ids_in_both:
            row_new = casted_new.filter(pl.col(pks[0]) == id_val).to_dicts()[0]
            row_db = casted_db.filter(pl.col(pks[0]) == id_val).to_dicts()[0]

            # Compare casted values
            is_identical = True
            for key in row_new:
                if key in row_db and key not in pks:
                    # Both None is considered equal
                    if row_new[key] is None and row_db[key] is None:
                        continue
                    # One None and one not None is not equal
                    if row_new[key] is None or row_db[key] is None:
                        is_identical = False
                        break
                    # For dates and datetimes, compare by date part only
                    if isinstance(row_new[key], (date, datetime)) and isinstance(row_db[key], (date, datetime)):
                        if isinstance(row_new[key], datetime):
                            date_new = row_new[key].date()
                        else:
                            date_new = row_new[key]

                        if isinstance(row_db[key], datetime):
                            date_db = row_db[key].date()
                        else:
                            date_db = row_db[key]

                        if date_new != date_db:
                            is_identical = False
                            break
                    # For other types, compare values directly
                    elif row_new[key] != row_db[key]:
                        is_identical = False
                        break

            if is_identical:
                equals_ids.append(id_val)
            else:
                update_ids.append(id_val)

        # Create result DataFrames using original data
        equals_df = original_df_new.filter(pl.col(pks[0]).is_in(equals_ids)) if equals_ids else pl.DataFrame(schema=original_df_new.schema)
        update_df = original_df_new.filter(pl.col(pks[0]).is_in(update_ids)) if update_ids else pl.DataFrame(schema=original_df_new.schema)

        return insert_df, equals_df, update_df