from flask import Flask ,render_template,jsonify,request,url_for
import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder
import pandas as pd

## import labelencoder as le
app=Flask(__name__)
submodel=pickle.load(open("subpoppred.pkl","rb"))
heimodel=pickle.load(open("predheight.pkl","rb"))
pv=[]


@app.route('/height')
def height():
    return render_template("height.html")
@app.route('/', methods=['GET','POST'])
def predict():
   
    df=pd.read_csv("data/dataprediction.csv")
    y=df["Subpopulation"]
    le = LabelEncoder()
    le.fit_transform(y)
    # Get the input data from the request
    nucleotides=['A','C','G','T']
    new_seq = request.form.get('sequence')
    h=''
    subpop=''
    if new_seq:
    # Convert the input sequence to one-hot encoding
        one_hot1 = np.zeros((1, len(new_seq) * len(nucleotides)))
        for i, nuc in enumerate(new_seq):
            index = nucleotides.index(nuc)
            one_hot1[0, i*len(nucleotides) + index] = 1

    # Perform the prediction using the trained model
        subpop1 = submodel.predict(one_hot1)
        subpop = le.inverse_transform(subpop1)[0]
        inp=np.concatenate((one_hot1,subpop1.reshape(-1,1)),axis=1)
        h=heimodel.predict(inp)

  
    list1=[]
    list1.append(str(new_seq))
    list1.append(h)
    list1.append(str(subpop))
    pv.append(list1)
  
    return render_template('index.html',list2=pv,Seq=new_seq,h=h,subpop=subpop)



if __name__=="__main__":
    app.run(debug=True)
    
