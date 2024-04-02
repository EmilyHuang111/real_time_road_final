import os  #type: ignore
import sqlite3 #Datebase
import hashlib
import tkinter as tk
from tkinter import Toplevel, messagebox
from tkinter import ttk #Imported for updating GUI style
import requests #imported for restful api calls
from login import *
from register import *

#Creat database instance
connection = sqlite3.connect("database.db")
cursor = connection.cursor()
#Create tables in the database for first name, last name, username, password and salt
cursor.execute('''CREATE TABLE IF NOT EXISTS user_information (
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  first_name TEXT,
                  last_name TEXT,
                  username TEXT,
                  password_hash BLOB,
                  salt BLOB)''')
connection.commit()
connection.close()

#Function to exit the program
def exit_app(): 
  root.quit()

# Create starting login/register window 
# Create a width for all the buttons
button_width = 10
frm = ttk.Frame(root, padding=10)
frm.grid()
ttk.Label(frm, text="Login with an existing account:").grid(row=0,column=0,padx=20,pady=10)
login_button = ttk.Button(frm, text="Login", command=login_menu,width=button_width)
login_button.grid(row=0, column=1, padx=20, pady=10)
# Creat a button to register a new account
ttk.Label(frm, text="Or register a new account:").grid(row=1,column=0,padx=20,pady=10)
register_button = ttk.Button(frm, text="Register", command=register_menu,width=button_width)
register_button.grid(row=1, column=1, padx=20, pady=10)
# Create Exit button to exit the program
exit_button = ttk.Button(frm, text="Exit", command=exit_app)
exit_button.grid(row=2, column=0, columnspan=2, padx=20, pady=10)

root.mainloop()

#https://github.com/Dt-Pham/Advanced-Lane-Lines/blob/master/LaneLines.py
#https://www.youtube.com/watch?v=iRTuCYx6quQ
#https://github.com/kemfic/Curved-Lane-Lines/blob/master/P4.ipynb
#https://www.hackster.io/kemfic/curved-lane-detection-34f771
