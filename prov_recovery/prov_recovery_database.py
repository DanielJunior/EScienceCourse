import sqlite3

class ProvRecoveryDatabase():
    def __init__(self):
        pass

    CREATE_ENTITY_TABLE = """
        CREATE TABLE IF NOT EXISTS entity (
        id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        modified_date TIMESTAMP
        ); """

    CREATE_WAS_DERIVED_TABLE = """
            CREATE TABLE IF NOT EXISTS was_derived (
            id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
            entity_id INTEGER NOT NULL,
            ancestor_id INTEGER
            ); """

    INSERT_INTO_ENTITY = """
            INSERT INTO entity (name, modified_date)
            VALUES (?, ?)"""

    INSERT_INTO_WAS_DERIVED = """
                INSERT INTO was_derived (entity_id, ancestor_id)
                VALUES (?, ?)"""

    def create_database(self, cursor):
        cursor.execute(ProvRecoveryDatabase.CREATE_ENTITY_TABLE)
        cursor.execute(ProvRecoveryDatabase.CREATE_WAS_DERIVED_TABLE)
        self.commit()

    def open_database(self):
        self.conn = sqlite3.connect('database/prov_recovery.db')
        self.create_database(self.conn.cursor())
        return self.conn.cursor()

    def close_database(self):
        self.conn.close()

    def insert_into_entity(self, name, modified_date):
        cursor = self.open_database()
        cursor.execute(ProvRecoveryDatabase.INSERT_INTO_ENTITY, (name, modified_date))
        self.commit()

    def insert_into_was_derived(self, entity, ancestor):
        cursor = self.open_database()
        if not ancestor == None:
            sql = "SELECT id FROM entity WHERE name LIKE '" + str(entity) + "' OR name LIKE '" + str(
                ancestor) + "' ORDER BY id;"
            cursor.execute(sql)
            result = cursor.fetchall()
            ancestor_id = result[0][0]
            entity_id = result[1][0]
            cursor.execute(ProvRecoveryDatabase.INSERT_INTO_WAS_DERIVED, (entity_id, ancestor_id))
        else:
            sql = "SELECT id FROM entity WHERE name LIKE '" + str(entity) + "';"
            cursor.execute(sql)
            ids = cursor.fetchall()[0]
            cursor.execute(ProvRecoveryDatabase.INSERT_INTO_WAS_DERIVED, (ids[0], "NULL"))
        self.commit()

    def commit(self):
        self.conn.commit()
