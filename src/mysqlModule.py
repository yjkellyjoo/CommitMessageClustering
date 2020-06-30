import mysql.connector as sql
from mysql.connector import Error


def create_connection(host_name, user_name, user_password, user_port, db_name):
    connection = None
    try:
        connection = sql.connect(
            host=host_name,
            user=user_name,
            passwd=user_password,
            port=user_port,
            database=db_name
        )
        print("Connection to MySQL DB successful")
    except Error as e:
        print(f"The error '{e}' occurred")

    return connection

def execute_read_query(connection, query):
    cursor = connection.cursor()
    result = None
    try:
        cursor.execute(query)
        result = cursor.fetchall()
        return result
    except Error as e:
        print(f"The error '{e}' occurred")

def execute_read_one_query(connection, query):
    cursor = connection.cursor()
    result = None
    try:
        cursor.execute(query)
        result = cursor.fetchone()
        return result
    except Error as e:
        print(f"The error '{e}' occurred")

def execute_update_query(connection, query):
    cursor = connection.cursor()
    try:
        cursor.execute(query)
        connection.commit()
    except Error as e:
        print(f"The error '{e}' occurred")

