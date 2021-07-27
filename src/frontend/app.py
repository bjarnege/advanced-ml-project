from flask import Flask, render_template, request, redirect, url_for
import requests
import pickle
import jinja2

with open('results.pkl', 'rb') as f:
    data = pickle.load(f)

# define function to retreive all data from our API
def find_papers(input_link, pipeline):
    url = f"http://127.0.0.1:12345/api?url={input_link}&pipeline={pipeline}"
    return {input_link: requests.get(url).json()}

# app = Flask(__name__) creates an instance of the Flask class called app. 
# the first argument is the name of the module or package (in this case Flask). 
# we are passing the argument __name__ to the constructor as the name of the application package. 
# it is used by Flask to find static assets, templates and so on.
app = Flask(__name__)

# disabling caching of this app
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

input_link = False
error = False

@app.route("/")
def index():
    return render_template('index_2.html')

@app.route("/find_paper")
def find_paper():
    try:
        return render_template('find_paper.html', data=data, input_link=input_link, error=error)
    except jinja2.exceptions.UndefinedError:
        return render_template('find_paper.html', data=data, input_link=False, error="Your URL does not work. Please try again.")

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
# If the script is imported from another script, the script keeps it given name (e.g. app.py). 
# In our case we are executing the script. Therefore, __name__ will be equal to "__main__". 
# That means the if conditional statement is satisfied and the app.run() method will be executed.
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)