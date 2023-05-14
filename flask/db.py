import pymysql

class Database:
    def __init__(self,host,port,user,password,database_name):
        # MySQL connection
        self.connect=pymysql.connect(host=host,port=port,user=user,password=password,db=database_name,charset='utf8')
        self.cursor=self.connect.cursor()

    def getData(self, table_name, targets='*',option=None):
        if option == None:
            sql = f"select {targets} from {table_name};"
        else:
            sql = f"select {targets} from {table_name} where {option};"

        self.cursor.execute(sql)
        rows = self.cursor.fetchall()
        self.connect.commit()

        return rows
    
    def insData(self, table_name, columns, values):
        options=''
        for t in values:
            options+='%s,'
        options=options[:-1]

        sql = f"INSERT INTO {table_name} ({','.join(columns)})" \
              f"VALUES ({options})"
        #print(sql,values)
        self.cursor.execute(sql, values)
        self.connect.commit()
        
    def delData(self, table_name, id):
        sql = f'delete from {table_name} where ID=%s'
        self.cursor.execute(sql, id)
        self.connect.commit()
        
    def db_disconnect(self):
        self.connect.close()

    def __del__(self):
        self.db_disconnect()