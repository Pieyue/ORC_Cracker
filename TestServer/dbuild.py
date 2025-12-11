import mysql.connector
from contextlib import contextmanager

class Database:
    def __init__(self, host, user, password, database, port=3306):
        self.host=host
        self.user = user
        self.password = password
        self.database = database
        self.port = port

    def connection(self):
        return mysql.connector.connect(
            host=self.host,
            database=self.database,
            user=self.user,
            password=self.password
        )

    # 将生成器函数变为上下文管理器
    @contextmanager
    def get_cursor(self):
        # 游标是数据库连接中用于执行SQL的对象
        conn = self.connection()
        cursor = conn.cursor()  # 从连接创建游标用于执行SQL
        try:
            yield cursor    # 产出的cursor作为上下文管理器
            conn.commit()   # with结束后提交事务
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            cursor.close()
            conn.close()

    def execute_sql(self, sql:str, params=None):
        with self.get_cursor() as cursor:
            cursor.execute(sql, params)
            # startswith()判断是否以某个字符串开头
            if sql.strip().lower().startswith('select'):
                # 只有select语句会返回，所以只在此情况返回结果
                return cursor.fetchall()    # fetchall()获取所有返回结果