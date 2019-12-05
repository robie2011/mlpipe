import sqlite3
from enum import Enum

# todo: get env var
from dataclasses import dataclass

DB_STORE = "/tmp/mlpipe/db.sqlite3"
c = object

def auto_initialize():
    conn = sqlite3.connect(DB_STORE)
    c = conn.cursor()
    init_statements = ["""
        create table if not exists meta (
        id string, 
        insert_date datetime default current_timestamp, 
        category TEXT, 
        name TEXT)""",

        "CREATE UNIQUE INDEX IF NOT EXISTS ix_meta_id on meta (id)",
        "CREATE UNIQUE INDEX IF NOT EXISTS ix_meta_category_name on meta (category, name)",
        ]

    for statement in init_statements:
        c.execute(statement)

    conn.close()


class Categories(Enum):
    data = "data"
    training = "training"


@dataclass
class MetaInfo:
    uuid: str
    insert_datetime_local: str
    category: Categories
    name: str


def write(category: Categories, name: str, identifier: str):
    with sqlite3.connect(DB_STORE) as conn:
        cursor: sqlite3.Cursor = conn.cursor()
        cursor.execute("""
        INSERT INTO meta (id, category, name)
        VALUES (?, ?, ?)
        """, (identifier, category.value, name))
        conn.commit()
        return True


def list_category(category: Categories):
    with sqlite3.connect(DB_STORE) as conn:
        cursor: sqlite3.Cursor = conn.cursor()
        cursor.execute(
            "SELECT id, datetime(insert_date, 'localtime'), category, name FROM meta where category=?",
            (category.value,))

        return [
            MetaInfo(
                uuid=x[0],
                insert_datetime_local=x[1],
                category=Categories(x[2]),
                name=x[3])
            for x in cursor.fetchall()]


# def get(category: Categories, name_or_id: str):
#     params = (category.value, f"%{name_or_id}%", f"{name_or_id}%")
#     c.execute("SELECT id FROM meta where category=? and (name like ? OR id like ?)", params)
#     return c.fetchone()[0]


auto_initialize()
