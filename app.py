import numpy as np
from flask import Flask,jsonify,render_template,request,redirect,url_for
import pickle

app=Flask(__name__)
rfcmodel=pickle.load(open('rfc_model.pkl','rb'))
logregmodel=pickle.load(open('log_model.pkl','rb'))

@app.route('/')
def home():
    return render_template('hero.html')
@app.route('/project_v1.html')
def proj():
    return render_template('project_v1.html')
@app.route("/predict", methods=['GET',"POST"])
def submit():
    if request.method=="POST":
        loan_amt=float(request.form['Loan Amount'])
        fund_amt=float(request.form['Funded Amount'])
        fund_amt_inv=float(request.form['Funded Amount Investor'])
        int_rate=float(request.form['Interest rate'])

        grade=request.form['Grade']
        sub_grade=request.form['Sub grade']
        emp_dur=request.form['Employment duration']
        home_own=float(request.form['Home ownership'])

        ver_stat=request.form['Verification status']
        debit_to_income=float(request.form['Debit to income'])
        open_ac=int(request.form['Open Account'])
        rev_uti=float(request.form['Revolving Utilities'])
        
        tot_ac=int(request.form['Total Accounts'])
        init_list_stat=request.form['Initial List Status']
        l_w_p=int(request.form['Last week Pay'])
        tot_rev_cred_limit=int(request.form['Total Revolving Credit Limit'])
        user_choice=request.form['Model']
        Grade={'A':[0,0,0], 'B':[0,0,1], 'C':[0,1,0], 'D':[0,1,1],
               'E':[1,0,0], 'F':[1,0,1], 'G':[1,1,0]}
        
        Sub_grade={'A1':[0,0,0,0,0,0], 'A2':[0,0,0,0,0,1], 'A3':[0,0,0,0,1,0],
                   'A4':[0,0,0,0,1,1], 'A5':[0,0,0,1,0,0],
                   
                   'B1':[0,0,0,1,0,1], 'B2':[0,0,0,1,1,0], 'B3':[0,0,0,1,1,1],
                   'B4':[0,0,1,0,0,0], 'B5':[0,0,1,0,0,1],
                   
                   'C1':[0,0,1,0,1,0], 'C2':[0,0,1,0,1,1], 'C3':[0,0,1,1,0,0],
                   'C4':[0,0,1,1,0,1], 'C5':[0,0,1,1,1,0],
                   
                   'D1':[0,0,1,1,1,1], 'D2':[0,1,0,0,0,0], 'D3':[0,1,0,0,0,1],
                   'D4':[0,1,0,0,1,0], 'D5':[0,1,0,0,1,1],
                   
                   'E1':[0,1,0,1,0,0], 'E2':[0,1,0,1,0,1], 'E3':[0,1,0,1,1,0],
                   'E4':[0,1,0,1,1,1], 'E5':[0,1,1,0,0,0],
                   
                   'F1':[0,1,1,0,0,1], 'F2':[0,1,1,0,1,0], 'F3':[0,1,1,0,1,1],
                   'F4':[0,1,1,1,0,0], 'F5':[0,1,1,1,0,1],
                   
                   'G1':[0,1,1,1,1,0], 'G2':[0,1,1,1,1,1], 'G3':[1,0,0,0,0,0],
                   'G4':[1,0,0,0,0,1], 'G5':[1,0,0,0,1,0]}

        Emp_dur={'Mortgage':[0,0], 'Rent':[0,1], 'Own':[1,0]}

        Ver_stat={'Not Verified':[0,0], 'Source Verified':[0,1],
                  'Verified':[1,0]}

        Init_list_stat={'Waiting':[1,0], 'Forwarded':[0,1]}
        inp=[]
        inp.append(loan_amt)
        inp.append(fund_amt)
        inp.append(fund_amt_inv)
        inp.append(int_rate)
        temp=Grade[grade]
        inp.extend(temp)
        temp=Sub_grade[sub_grade][1:]
        inp.extend(temp)  
        temp=Emp_dur[emp_dur]
        inp.extend(temp) 
        inp.append(home_own)
        temp=Ver_stat[ver_stat]
        inp.extend(temp)   
        inp.append(debit_to_income)
        inp.append(open_ac)
        inp.append(rev_uti)
        inp.append(tot_ac)
        temp=Init_list_stat[init_list_stat]
        inp.extend(temp)  
        inp.append(l_w_p)
        inp.append(tot_rev_cred_limit)
        final_inp=[np.array(inp)]
        if user_choice=='rfc':
            pred=rfcmodel.predict(final_inp)
        else:
            pred=logregmodel.predict(final_inp)
        #out=round(pred[0],1)
        if pred==0:
            return render_template('project_v1.html',pred_text=" The concerned party is not a defaulter".format(pred[0]))
        else:
            return render_template('project_v1.html',pred_text=" The concerned party is likely a defaulter".format(pred[0]))
        
    else:
        return render_template('project_v1.html')

if __name__ == "__main__":
    app.run(debug=True)

