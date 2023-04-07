import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from flask import  render_template
import sys, os

from sklearn.metrics.pairwise import cosine_similarity

current_directory = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.dirname(current_directory))
from pathlib import Path
app = Flask(__name__,
        template_folder='D:/Convertedin/temp'
        )
# app = Flask(__name__,template_folder='../temp')
df = pd.read_csv('Dataset.csv')
df =df.dropna()

def recommeditems_user(target_user):

  target=int(target_user)
  customer_matrix = df.pivot_table(
    index='CustomerID',
    columns='StockCode',
    values='Quantity',
    aggfunc='sum')
  
  customer_matrix = customer_matrix.applymap(lambda x: 1 if x > 0 else 0) 
  user_item_matrix = csr_matrix(customer_matrix.values)

  model_knn = NearestNeighbors(metric = 'cosine', algorithm = 'brute')
  model_knn.fit(user_item_matrix)

  index_pos = customer_matrix.index.get_loc(target)

  target=customer_matrix.iloc[index_pos,:].values.reshape(1,-1)


  distances, indices = model_knn.kneighbors(target, n_neighbors=5)


  similar_users=[]
  distance=[]
  for i in range(0, len(distances.flatten())):
      if i != 0:
        similar_users.append(customer_matrix.index[indices.flatten()[i]])
        distance.append(distances.flatten()[i]) 


  m=pd.Series(similar_users,name='CustomerID')
  d=pd.Series(distance,name='distance')
  recommend = pd.concat([m,d], axis=1)
  recommend = recommend.sort_values('distance',ascending=False)

  # print('Similar users for customer ID: {0} \n'.format(customer_matrix.index[random_cutomer]))
  # for i in range(0,recommend.shape[0]):
  #     print(' customer_id: {1}, with distance of {2}'.format(i, recommend["CustomerID"].iloc[i], recommend["distance"].iloc[i]))

  items=[]
  for t in similar_users:
      items += customer_matrix.loc[t,:][customer_matrix.loc[t,:] > 0].index.tolist()

  diff_items = list(set(items) - set(customer_matrix.index[index_pos] [customer_matrix.index[index_pos] > 0]))

  # recommend the 10 top items to the target user
  N = 10
  recommended_items = pd.Series(0, index=diff_items).sort_values(ascending=False)[:N].index.tolist()

  # print('Recommended items for customer ID:', customer_matrix.iloc[index_pos], ':\n', recommended_items)

  Recommended_items = pd.DataFrame(columns=['user-user StockCode', 'Description'])

  for i, stockcode in enumerate(recommended_items):
      desc = df.loc[df['StockCode'] == stockcode, 'Description'].values[0]
      Recommended_items.loc[i, 'user-user StockCode'] = stockcode  
      Recommended_items.loc[i, 'Description'] = desc 

  return Recommended_items

def recommeditems_item(itemid):
    customer_matrix = df.pivot_table(
    index='CustomerID',
    columns='StockCode',
    values='Quantity',
    aggfunc='sum')

    customer_matrix = customer_matrix.applymap(lambda x: 1 if x > 0 else 0) 
    item_matrix=customer_matrix.T
    itemid=str(itemid)

    index_pos = item_matrix.index.get_loc(itemid)


    similarity = cosine_similarity(item_matrix) 
    item_similarity= similarity[index_pos]

    sorted_similarity= item_similarity.argsort()[::-1]

    num_similar_items = 10
    similar_items = []
    for idx in sorted_similarity:
        if item_matrix.index[idx] != str(itemid): #check the other similar items
            similar_items.append(item_matrix.index[idx])
            if len(similar_items) == num_similar_items:
                break
    
    stockcode_desc = df[df['StockCode'] == itemid]['Description'].iloc[0]
    # print('Recommended items for StackCode:', itemid,"with decription",stockcode_desc,":\n",similar_items )

    items_Recomendation = pd.DataFrame(columns=['item-item StockCode', 'Description'])

    for i, stockcode in enumerate(similar_items):
        desc = df[df['StockCode'] == stockcode]['Description'].iloc[0]
        items_Recomendation.loc[i, 'item-item StockCode'] = stockcode  
        items_Recomendation.loc[i, 'Description'] = desc

    return items_Recomendation


@app.route('/')
def homepage():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    


    user = int(request.form['userID'])
    output1 = recommeditems_user(user)
    table1 = output1.to_html(index=False, border=1)
    
    item = str(request.form['itemID'])
    output2 = recommeditems_item(item)
    table2 = output2.to_html(index=False, border=1)
    return render_template("index.html", table1=table1, table2=table2)

 


if __name__ == "__main__":
    app.run('0.0.0.0', 5000, debug=True)