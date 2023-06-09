from new_conditions import compare_word
import pandas as pd
import numpy as np
from flask import Flask, jsonify,request, render_template, redirect, url_for, session
import fasttext
import fasttext.util
import collections
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import json
import sys
from collections import defaultdict
from text_summerization import summarize_sentences


# Create a Flask app
app = Flask(__name__, static_url_path='/static')

# Define a route and its corresponding view function

@app.route('/search')
def searchpage():
    return render_template("search.html")

@app.route('/searchresult', methods=['POST'])
def search_condition():
    word = request.data.decode('utf-8')
    medicine = request.form.get("DrugName")
    ft = fasttext.load_model('cc.en.300.bin')

    df = pd.read_csv("webmd.csv")
    conditon = df[df['Drug']== medicine].Condition.unique()
    
    summary = summarize_sentences(conditon)
        
    words_vectors={w:ft.get_sentence_vector(w) for  w in conditon}

    top_search = compare_word(summary, words_vectors)

    ts = collections.Counter(top_search).most_common()[:10]

    json_dict = {}
    for i,j in ts:
        json_dict[i]=str(j)
        # return jsonify(json_dict)
    print(json_dict)
    
    return render_template("searchresult.html",data = json_dict)

@app.route('/finalresult', methods=['POST'])
def finalresult():
    df = pd.read_csv("webmd.csv")
    condition = request.form.get("condition")
    data = df[df['Condition']== condition]
    drugs = data.Drug.unique()

    l = {}
    for drug in drugs:
        filtered_df = df[(df['Drug'] == drug) & (df['Condition']== condition)]
        l[drug] = (round(filtered_df.Effectiveness.mean(),2))

    l = dict(sorted(l.items(),key=lambda item : item[1],reverse=True)[:5])    
    obj = {}
    obj["drugs"] = list(l.keys())
    obj["labels"] = list(l.values())

    df = pd.read_csv("webmd.csv")

    df = df[(df['Condition']==condition)]

   
    # drugs = df.Drug.unique()

    drug_reviews_list =defaultdict(list)
    for d in list(l.keys()):
        temp_df = df[df['Drug']==str(d)].Reviews.tolist()
        text = " ".join(temp_df)
        if len(text)>20:
            wordcloud = WordCloud().generate(text)
            all_words = wordcloud.words_
            drug_reviews_list[d].append(all_words)
        


    data = []
    for drug in drug_reviews_list:
        for i in drug_reviews_list[drug]:
            try:
                sorted_list = sorted(i, key=lambda x: x[1])
                word_cloud_list = sorted_list[:20]
                for j in word_cloud_list:
                    a = {'x': '', 'value':'','category':''}
                    a['x'] = str(j)
                    a['value'] = str(i[j])
                    a["category"]=drug
                    data.append(a)
            except:
                print("")        
    obj['wordcloud'] = data

    data = []
    data = collections.defaultdict(list)
    
    for drug in list(l.keys()):
        temp = []
        filtered_df = df[(df['Drug'] == drug) & (df['Condition']== condition)]
        temp.append(round(filtered_df.Effectiveness.mean(),2))
        temp.append(round(filtered_df.EaseofUse.mean(),2))
        temp.append(round(filtered_df.Satisfaction.mean(),2))
        data[drug].append(temp)
   
    def flatten_3d_list(lst):
        flattened_list = []
        for sublist in lst:
            if isinstance(sublist, list):
                flattened_list.extend(flatten_3d_list(sublist))
            else:
                flattened_list.append(sublist)
        return flattened_list
    l = flatten_3d_list(list(data.values()))
            
    obj['drugtitle'] = list(data.keys())
    obj['graphvalues'] = l
    return render_template("finalresult.html",data = json.dumps(obj))

@app.route('/', methods=['GET'])
def homepage():
    return render_template("index.html")

@app.route('/generatecloud', methods=['GET'])
def generatecloud():
    df = pd.read_csv("webmd.csv")
    
generatecloud()


# # Run the app
if __name__ == '__main__':
    try:
        app.run(debug=True ,port=3000,use_reloader=False)

    except KeyboardInterrupt:
        # Shut down the server on keyboard interrupt
        print('Server shutting down...')
        sys.exit()







