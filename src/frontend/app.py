from flask import Flask, render_template, request, redirect, url_for
import requests
import pickle
import jinja2

with open('results.pkl', 'rb') as f:
    data = pickle.load(f)

# define function to retreive alle data from our API
# def get_data():
    #url =  "http://0.0.0.0:12345/recommend"
    #data = requests.get(url).json()
    #return data

def find_papers(input_link, pipeline):
    url = f"0.0.0.0:12345/api?url={input_link}&pipeline={pipeline}"
    #return requests.get(url).content.decode()

# app = Flask(__name__) creates an instance of the Flask class called app. 
# the first argument is the name of the module or package (in this case Flask). 
# we are passing the argument __name__ to the constructor as the name of the application package. 
# it is used by Flask to find static assets, templates and so on.
app = Flask(__name__)

# disabling caching of this app
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

# we set the Track Modifications to True so thatFlask-SQLAlchemy will track modifications of objects and emit signals. 
# the default is None, which enables tracking but issues a warning that it will be disabled by default in the future. 
# this requires extra memory and should be disabled if not needed.
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = True

# this code registers the index() view function as a handler for the root URL of the application. 
# everytime the application receives a request where the path is "/" the index() function will be invoked and the return the index.html template.
#data = False
input_link = False
error = False

@app.route("/")
def index():
    return render_template('index_2.html')

# This function displays all books that are currently in the database as well as their availability status. 
# The data is extracted via the pandas.read_sql function and saved in MyBook, which is used in the newbook.html to display all books. 
# It returns a table that includes every book in the database as well as the correct author_id. 
# The join is performed to add a column that contains the isbn of books that are borrowed and that contains NONE when the book is not borrowed.
# Converting the isbn to an int is necessary to be able to alter books. 
@app.route("/find_paper")
def find_paper():
    try:
        return render_template('find_paper.html', data=data, input_link=input_link, error=error)
    except jinja2.exceptions.UndefinedError:
        return render_template('find_paper.html', data=data, input_link=False, error="Your URL does not work. Please try again.")

# This function is used to add new books to the database. 
# The HTTP methods post and get are used when accessing URLs. 
# GET is used to request data from a specified resource.
# POST is used to send data to a server to create/update a resource. In this case we create a new book with all its attributes. 
@app.route("/find_paper_end", methods=['GET', 'POST'])
def find_paper_end():
    pipeline = ",".join(request.form.getlist("pipeline"))
    global data
    global input_link
    input_link = request.form['input_link']
    #data = find_papers(input_link, pipeline)
    return redirect(url_for('find_paper'))

# This function returns the contact.html page containing the contact details from our team.
@app.route("/team")
def team():
    return render_template('contact.html')

# Python assigns the name "__main__" to the script when the script is executed. 
# If the script is imported from another script, the script keeps it given name (e.g. library.py). 
# In our case we are executing the script. Therefore, __name__ will be equal to "__main__". 
# That means the if conditional statement is satisfied and the app.run() method will be executed.
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)