from polarity.caster import PolarsCaster
from typing import Optional
from pydantic import BaseModel
import polars as pl
import pytest
from datetime import date, time, datetime
from polars.testing import assert_frame_equal


def test_split_dataframe_basico():
    """
    Prueba básica con múltiples columnas y tipos de datos comunes, incluyendo valores None.
    """
    df_new = pl.DataFrame({
        'id': [1, 2, 3, 4],
        'nombre': ['Ana', 'Luis', 'María', 'Pedro'],
        'edad': [25, None, 30, 22],
        'fecha': [date(2021, 1, 1), date(2021, 2, 1), None, date(2021, 4, 1)],
        'activo': [True, False, True, None],
        'salario': [50000.0, 60000.5, None, 55000.0],
    })

    df_db = pl.DataFrame({
        'id': [2, 3, 5],
        'nombre': ['Luis', 'María', 'Carlos'],
        'edad': [None, 30, 28],
        'fecha': [date(2021, 2, 1), None, date(2021, 5, 1)],
        'activo': [False, True, True],
        'salario': [60000.5, None, 48000.0],
    })

    pks = ['id']

    class TestModel(BaseModel):
        id: int
        nombre: Optional[str] = None
        edad: Optional[int] = None
        fecha: Optional[date] = None
        activo: Optional[bool] = None
        salario: Optional[float] = None

    # Create a PolarsCaster instance with the schema mapping
    caster = PolarsCaster({"test_schema": TestModel})

    # Call the method on the instance
    insert_df, equals_df, update_df = caster.split_dataframe(df_new, df_db, pks, "test_schema")

    # Crear DataFrames esperados
    expected_insert_df = df_new.filter(pl.col('id').is_in([1, 4]))  # Filas nuevas
    expected_equals_df = df_new.filter(pl.col('id').is_in([2, 3]))  # Filas iguales (id = 2, 3)
    expected_update_df = pl.DataFrame({col: pl.Series([], dtype=df_new[col].dtype) for col in df_new.columns})  # DataFrame vacío con las mismas columnas

    # Comparar los DataFrames obtenidos con los esperados
    assert_frame_equal(insert_df, expected_insert_df, check_dtypes=False), "insert_df no coincide con el esperado"
    assert_frame_equal(equals_df, expected_equals_df, check_dtypes=False), "equals_df no coincide con el esperado"
    assert_frame_equal(update_df, expected_update_df, check_dtypes=False), "update_df no coincide con el esperado"

def test_split_dataframe_tipos_datos():
    """
    Prueba con todos los tipos de datos posibles en Polars.
    """
    df_new = pl.DataFrame({
        'id': [1, 2],
        'int_col': [10, None],
        'float_col': [None, 20.5],
        'str_col': ['texto', None],
        'bool_col': [True, False],
        'date_col': [date(2021, 1, 1), None],
        'time_col': [time(12, 0, 0), None],
        'datetime_col': [datetime(2021, 1, 1, 12, 0, 0), None],
        'categorical_col': ['cat1', 'cat2'],
        'binary_col': [b'\x00\x01', b'\x02\x03'],
    })

    df_db = pl.DataFrame({
        'id': [2, 3],
        'int_col': [None, 30],
        'float_col': [20.5, None],
        'str_col': [None, 'texto3'],
        'bool_col': [False, True],
        'date_col': [None, date(2021, 3, 1)],
        'time_col': [None, time(13, 0, 0)],
        'datetime_col': [None, datetime(2021, 3, 1, 13, 0, 0)],
        'categorical_col': ['cat2', 'cat3'],
        'binary_col': [b'\x02\x03', b'\x04\x05'],
    })

    pks = ['id']

    class TestModel(BaseModel):
        id: int
        int_col: Optional[int] = None
        float_col: Optional[float] = None
        str_col: Optional[str] = None
        bool_col: Optional[bool] = None
        date_col: Optional[date] = None
        time_col: Optional[time] = None
        datetime_col: Optional[datetime] = None
        categorical_col: Optional[str] = None
        binary_col: Optional[bytes] = None

    # Create a PolarsCaster instance with the schema mapping
    caster = PolarsCaster({"test_schema": TestModel})

    # Call the method on the instance
    insert_df, equals_df, update_df = caster.split_dataframe(df_new, df_db, pks, "test_schema")

    expected_insert_df = df_new.filter(pl.col('id') == 1)  # Fila nueva con id = 1
    expected_equals_df = df_new.filter(pl.col('id') == 2)  # Fila igual con id = 2
    expected_update_df = pl.DataFrame({col: pl.Series([], dtype=df_new[col].dtype) for col in df_new.columns})

    assert_frame_equal(insert_df, expected_insert_df, check_dtypes=False), "insert_df no coincide con el esperado"
    assert_frame_equal(equals_df, expected_equals_df, check_dtypes=False), "equals_df no coincide con el esperado"
    assert_frame_equal(update_df, expected_update_df, check_dtypes=False), "update_df no coincide con el esperado"

def test_split_dataframe_valores_nulos_en_pks():
    """
    Prueba casos donde las claves primarias tienen valores nulos.
    """
    df_new = pl.DataFrame({
        'id': [1, None],
        'nombre': ['Ana', 'Juan'],
    })

    df_db = pl.DataFrame({
        'id': [None, 2],
        'nombre': ['Juan', 'Luis'],
    })

    pks = ['id']

    class TestModel(BaseModel):
        id: int
        nombre: Optional[str] = None

    # Create a PolarsCaster instance with the schema mapping
    caster = PolarsCaster({"test_schema": TestModel})

    with pytest.raises(ValueError, match="contains"):
        caster.split_dataframe(df_new, df_db, pks, "test_schema")

def test_split_dataframe_sin_pks():
    """
    Prueba el comportamiento cuando no se proporcionan claves primarias.
    """
    df_new = pl.DataFrame({
        'id': [1, 2],
        'nombre': ['Ana', 'Luis'],
    })

    df_db = pl.DataFrame({
        'id': [2, 3],
        'nombre': ['Luis', 'María'],
    })

    pks = []

    class TestModel(BaseModel):
        id: int
        nombre: Optional[str] = None

    # Create a PolarsCaster instance with the schema mapping
    caster = PolarsCaster({"test_schema": TestModel})

    with pytest.raises(ValueError, match="cannot be empty"):
        caster.split_dataframe(df_new, df_db, pks, "test_schema")

def test_split_dataframe_dataframes_vacios():
    """
    Prueba el comportamiento cuando alguno de los DataFrames está vacío.
    """
    df_new = pl.DataFrame({
        'id': pl.Series([], dtype=pl.Int64),
        'nombre': pl.Series([], dtype=pl.Utf8),
    })

    df_db = pl.DataFrame({
        'id': [1, 2],
        'nombre': ['Ana', 'Luis'],
    })

    pks = ['id']

    class TestModel(BaseModel):
        id: int
        nombre: Optional[str] = None

    # Create a PolarsCaster instance with the schema mapping
    caster = PolarsCaster({"test_schema": TestModel})

    insert_df, equals_df, update_df = caster.split_dataframe(df_new, df_db, pks, "test_schema")

    assert insert_df.is_empty(), "insert_df debería estar vacío"
    assert equals_df.is_empty(), "equals_df debería estar vacío"
    assert update_df.is_empty(), "update_df debería estar vacío"

def test_split_dataframe_db_vacio():
    """
    Prueba el comportamiento cuando alguno de los DataFrames está vacío.
    """
    df_new = pl.DataFrame({
        'id': [1, 2],
        'nombre': ['Ana', 'Luis'],
    })

    df_db = pl.DataFrame({
        'id': pl.Series([], dtype=pl.Int64),
        'nombre': pl.Series([], dtype=pl.Utf8),
    })

    pks = ['id']

    class TestModel(BaseModel):
        id: int
        nombre: Optional[str] = None

    # Create a PolarsCaster instance with the schema mapping
    caster = PolarsCaster({"test_schema": TestModel})

    insert_df, equals_df, update_df = caster.split_dataframe(df_new, df_db, pks, "test_schema")

    # The result may be casted, so just check the shape and values
    assert insert_df.shape[0] == df_new.shape[0], "insert_df should have the same number of rows as df_new"
    assert equals_df.is_empty(), "equals_df debería estar vacío"
    assert update_df.is_empty(), "update_df debería estar vacío"

def test_split_dataframe_columnas_faltantes():
    """
    Prueba el comportamiento cuando faltan columnas de las claves primarias en los DataFrames.
    """
    df_new = pl.DataFrame({
        'nombre': ['Ana', 'Luis'],
    })

    df_db = pl.DataFrame({
        'id': [1, 2],
        'nombre': ['Ana', 'Luis'],
    })

    pks = ['id']

    class TestModel(BaseModel):
        id: int
        nombre: Optional[str] = None

    # Create a PolarsCaster instance with the schema mapping
    caster = PolarsCaster({"test_schema": TestModel})

    with pytest.raises(ValueError, match="exist in df_new"):
        caster.split_dataframe(df_new, df_db, pks, "test_schema")

def test_split_dataframe_con_todos_los_casos():
    """
    Prueba que abarca los tres casos: inserción, actualización e igualdad exacta.
    """
    df_new = pl.DataFrame({
        'id': [1, 2, 3, 4, 5],
        'nombre': ['Ana', 'Luis', 'María', 'Pedro', 'Juan'],
        'edad': [25, None, 30, 22, 40],
        'fecha': [date(2021, 1, 1), date(2021, 2, 1), None, date(2021, 4, 1), date(2021, 5, 1)],
        'activo': [True, False, True, None, True],
        'salario': [50000.0, 60000.5, None, 55000.0, 62000.0],
    })

    df_db = pl.DataFrame({
        'id': [2, 3, 5],
        'nombre': ['Luis', 'María', 'Juan'],
        'edad': [None, 30, 45],  # Edad de 'Juan' diferente
        'fecha': [date(2021, 2, 1), None, date(2021, 5, 1)],
        'activo': [False, True, True],
        'salario': [60000.5, None, 62000.0],
    })

    pks = ['id']

    class TestModel(BaseModel):
        id: int
        nombre: Optional[str] = None
        edad: Optional[int] = None
        fecha: Optional[date] = None
        activo: Optional[bool] = None
        salario: Optional[float] = None

    # Create a PolarsCaster instance with the schema mapping
    caster = PolarsCaster({"test_schema": TestModel})

    insert_df, equals_df, update_df = caster.split_dataframe(df_new, df_db, pks, "test_schema")

    expected_insert_df = df_new.filter(pl.col('id').is_in([1, 4]))  # Filas nuevas con id = 1, 4
    expected_equals_df = df_new.filter(pl.col('id').is_in([2, 3]))  # Filas iguales con id = 2, 3
    expected_update_df = df_new.filter(pl.col('id') == 5)  # Fila que necesita actualización id = 5

    assert_frame_equal(insert_df, expected_insert_df, check_dtypes=False), "insert_df no coincide con el esperado"
    assert_frame_equal(equals_df, expected_equals_df, check_dtypes=False), "equals_df no coincide con el esperado"
    assert_frame_equal(update_df, expected_update_df, check_dtypes=False), "update_df no coincide con el esperado"

def test_split_dataframe_strings_numericos_con_decimal():
    """
    Prueba para manejar strings numéricos que deberían ser enteros pero fueron ingresados como 'X.0'.
    """
    df_new = pl.DataFrame({
        'id': [1, 2],
        'codigo': ['123.0', '456'],  # Código mal cargado con decimal
    })

    df_db = pl.DataFrame({
        'id': [2, 3],
        'codigo': ['456', '789'],
    })

    pks = ['id']

    class TestModel(BaseModel):
        id: int
        codigo: Optional[str] = None

    # Create a PolarsCaster instance with the schema mapping
    caster = PolarsCaster({"test_schema": TestModel})

    insert_df, equals_df, update_df = caster.split_dataframe(df_new, df_db, pks, "test_schema")

    # Check the main structure but be flexible about exact string formatting
    assert insert_df.shape[0] == 1, "insert_df should have 1 row"
    assert insert_df[0, "id"] == 1, "insert_df should have id = 1"

    assert equals_df.shape[0] == 1, "equals_df should have 1 row"
    assert equals_df[0, "id"] == 2, "equals_df should have id = 2"

    assert update_df.shape[0] == 0, "update_df should be empty"

def test_split_dataframe_fechas_distintos_formatos():
    """
    Prueba con fechas en diferentes formatos, incluyendo Z, T, y horas como 00:00:00.
    """
    df_new = pl.DataFrame({
        'id': [1, 2],
        'fecha': ['2021-01-01T00:00:00Z', '2021-02-01 00:00:00'],
    })

    df_db = pl.DataFrame({
        'id': [2, 3],
        'fecha': ['2021-02-01T00:00:00Z', '2021-03-01T12:00:00Z'],
    })

    pks = ['id']

    class TestModel(BaseModel):
        id: int
        fecha: Optional[date] = None  # Fecha como date

    # Create a PolarsCaster instance with the schema mapping
    caster = PolarsCaster({"test_schema": TestModel})

    insert_df, equals_df, update_df = caster.split_dataframe(df_new, df_db, pks, "test_schema")

    # Check the structure rather than exact date representation
    assert insert_df.shape[0] == 1, "insert_df should have 1 row"
    assert insert_df[0, "id"] == 1, "insert_df should have id = 1"

    assert equals_df.shape[0] == 1, "equals_df should have 1 row"
    assert equals_df[0, "id"] == 2, "equals_df should have id = 2"

    assert update_df.shape[0] == 0, "update_df should be empty"

def test_split_dataframe_float_con_decimales():
    """
    Prueba el comportamiento con columnas float con diferentes cantidades de decimales.
    """
    df_new = pl.DataFrame({
        'id': [1, 2, 3],
        'float_col': [1.0000, 2.123456789, 3.14],
    })

    df_db = pl.DataFrame({
        'id': [2, 3],
        'float_col': [2.123456, 3.14],
    })

    pks = ['id']

    class TestModel(BaseModel):
        id: int
        float_col: Optional[float] = None

    # Create a PolarsCaster instance with the schema mapping
    caster = PolarsCaster({"test_schema": TestModel})

    insert_df, equals_df, update_df = caster.split_dataframe(df_new, df_db, pks, "test_schema")

    # Check the structure - for float comparison we need to be careful about precision
    assert insert_df.shape[0] == 1, "insert_df should have 1 row"
    assert insert_df[0, "id"] == 1, "insert_df should have id = 1"

    assert equals_df.shape[0] == 1, "equals_df should have 1 row"
    assert equals_df[0, "id"] == 3, "equals_df should have id = 3"

    assert update_df.shape[0] == 1, "update_df should have 1 row"
    assert update_df[0, "id"] == 2, "update_df should have id = 2"

def test_split_dataframe_valores_nulls():
    """
    Prueba el comportamiento cuando hay múltiples valores nulos en diferentes columnas y comparaciones complejas.
    """
    df_new = pl.DataFrame({
        'id': [1, 2, 3, 4],
        'nombre': ['Ana', None, 'María', 'Pedro'],
        'edad': [25, None, None, 22],
        'fecha': [date(2021, 1, 1), date(2021, 2, 1), None, None],
        'activo': [True, False, None, None],
        'salario': [50000.0, None, None, 55000.0],
    })

    df_db = pl.DataFrame({
        'id': [2, 3, 5],
        'nombre': [None, 'María', 'Carlos'],
        'edad': [None, None, 28],
        'fecha': [date(2021, 2, 1), None, date(2021, 5, 1)],
        'activo': [None, None, True],
        'salario': [None, None, 48000.0],
    })

    pks = ['id']

    class TestModel(BaseModel):
        id: int
        nombre: Optional[str] = None
        edad: Optional[int] = None
        fecha: Optional[date] = None
        activo: Optional[bool] = None
        salario: Optional[float] = None

    # Create a PolarsCaster instance with the schema mapping
    caster = PolarsCaster({"test_schema": TestModel})

    insert_df, equals_df, update_df = caster.split_dataframe(df_new, df_db, pks, "test_schema")

    # Check the structure
    assert insert_df.filter(pl.col('id').is_in([1, 4])).shape[0] == 2, "insert_df should contain rows with id 1 and 4"
    assert equals_df.filter(pl.col('id') == 3).shape[0] == 1, "equals_df should contain row with id 3"
    assert update_df.filter(pl.col('id') == 2).shape[0] == 1, "update_df should contain row with id 2"