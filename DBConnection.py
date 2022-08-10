import mysql.connector


class Db:
    def select(self, q):
        self.cnx = mysql.connector.connect(host="localhost",user="root",password="",database="spamdb")
        self.cur = self.cnx.cursor(dictionary=True)
        self.cur.execute(q)
        return self.cur.fetchall()

    def selectOne(self, q):
        self.cnx = mysql.connector.connect(host="localhost",user="root",password="",database="spamdb")
        self.cur = self.cnx.cursor(dictionary=True)
        self.cur.execute(q)
        return self.cur.fetchone()


    def insert(self, q):
        self.cnx = mysql.connector.connect(host="localhost",user="root",password="",database="spamdb")
        self.cur = self.cnx.cursor(dictionary=True)
        self.cur.execute(q)
        self.cnx.commit()
        return self.cur.lastrowid

    def update(self, q):
        self.cnx = mysql.connector.connect(host="localhost",user="root",password="",database="spamdb")
        self.cur = self.cnx.cursor(dictionary=True)
        self.cur.execute(q)
        self.cnx.commit()
        return self.cur.rowcount

    def delete(self, q):
        self.cnx = mysql.connector.connect(host="localhost",user="root",password="",database="spamdb")
        self.cur = self.cnx.cursor(dictionary=True)
        self.cur.execute(q)
        self.cnx.commit()
        return self.cur.rowcount

