from numpy.lib.function_base import percentile
from pandas.core.algorithms import mode
from pandas.io.parsers import CParserWrapper
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle


st.title("Car Price Prediction")

st.success('''The used car market is quite active in India.
 Therefore predicting car prices is highly variable. In this Project, 
 machine learning models are compared and chosen the best model for price prediction.''')


@st.cache()
def load_data(path):
    df= pd.read_csv(path)
    return df

df = load_data("dataset\cars_sampled.csv")

def load_model(path ='dataset\cars_sampled.csv'):
    with open(path,"rb") as file :
            model = pickle.load(file)
    st.sidebar.info("Model Loaded Sucessfully.")
    return model


if st.checkbox("About"):
    st.markdown("""The used car market is quite active in India.
 Therefore predicting car prices is highly variable. In this Project, 
 machine learning models are compared and chosen the best model for price prediction.
    """)
    st.markdown("""The training model was carried out by dividing the models by car brands instead of a single 
     big model training for both effective optimization processes and better use of computer power and time. 
     Approximately 100.000 data with 12 features were used in training. The machine learning models compared are:""")

    st.markdown('''
    * Linear Regression
* Ridge
* Lasso
* lastic Net
* K-Nearest Neighbors
* Random Forest
* XGBOOST
* Gradient Boosting Machine''')

    st.markdown('''First of all, the models were trained with default parameters. 
    Then, hyper-parameter optimization was applied to these models. The parameter values ​​and estimates 
    in both cases were recorded for later review.''')


if  st.checkbox("Make Prediction"):

    
    vehchile_type = st.selectbox("Vehchile Type",['limousine',
 'suv',
 'bus',
 'small car',
 'coupe',
 'station wagon',
 'others',
 'cabrio'])



    col3, col4 = st.beta_columns(2)

    with col3:
        gearbox = st.radio("Gearbox",["manual","automatic"])

    with col4:
        powerPS = st.number_input("Power PS")



    
    col5, col6 = st.beta_columns(2)

    with col5:
        modelname = st.selectbox("Model",['3er',
 'xc_reihe',
 'touran',
 'ibiza',
 'passat',
 'clk',
 'vectra',
 'octavia',
 'a_klasse',
 'astra',
 'yaris',
 'meriva',
 'others',
 'golf',
 '3_reihe',
 'carisma',
 'colt',
 '80',
 'panda',
 'micra',
 '156',
 'c_klasse',
 '1er',
 'e_klasse',
 'freelander',
 'polo',
 'a4',
 'forester',
 'cooper',
 '5er',
 'a3',
 'grand',
 'voyager',
 'fiesta',
 'clio',
 'slk',
 'sl',
 'x_reihe',
 'twingo',
 'fabia',
 'logan',
 '500',
 'punto',
 'berlingo',
 's_klasse',
 '2_reihe',
 'galaxy',
 'sharan',
 'agila',
 'captiva',
 'scenic',
 'mondeo',
 'omega',
 'a6',
 'fortwo',
 'scirocco',
 'transporter',
 'c4',
 'qashqai',
 'ka',
 'clubman',
 'insignia',
 '7er',
 'megane',
 'kadett',
 'corsa',
 'arosa',
 'focus',
 'z_reihe',
 'caddy',
 'aygo',
 'matiz',
 'combo',
 'corolla',
 '601',
 'c_max',
 'c_reihe',
 'beetle',
 'tigra',
 'tiguan',
 's_max',
 'toledo',
 'zafira',
 'a2',
 'primera',
 'transit',
 'a5',
 'tt',
 'citigo',
 'espace',
 'tucson',
 'accord',
 'rav',
 'i_reihe',
 'laguna',
 'roadster',
 'rio',
 'bora',
 'kuga',
 '147',
 'c1',
 'cuore',
 'civic',
 'mustang',
 'lupo',
 'c5',
 'mx_reihe',
 'cayenne',
 'justy',
 'avensis',
 'santa',
 'kaefer',
 'up',
 'yeti',
 'modus',
 'rx_reihe',
 'sprinter',
 'forfour',
 'cherokee',
 'cordoba',
 'v40',
 'navara',
 'v70',
 'q3',
 'sorento',
 'verso',
 'v_klasse',
 'doblo',
 '1_reihe',
 'fox',
 '100',
 '6_reihe',
 'altea',
 'almera',
 'impreza',
 'leon',
 'boxster',
 'calibra',
 'q5',
 '911',
 'x_trail',
 'v50',
 'bravo',
 'swift',
 'm_klasse',
 'jetta',
 'escort',
 'eos',
 'sandero',
 '4_reihe',
 '300c',
 'ceed',
 'stilo',
 'a1',
 'cl',
 'kangoo',
 'one',
 'alhambra',
 'pajero',
 'ducato',
 'vivaro',
 'touareg',
 'cr_reihe',
 '850',
 'q7',
 'duster',
 'cx_reihe',
 'seicento',
 'fusion',
 'b_klasse',
 'musa',
 'cc',
 'c3',
 'getz',
 'c2',
 '5_reihe',
 '900',
 'kalos',
 'ypsilon',
 'jazz',
 'ptcruiser',
 'sportage',
 'vito',
 'signum',
 'aveo',
 'exeo',
 'jimny',
 'note',
 'm_reihe',
 'auris',
 'juke',
 'picanto',
 'lancer',
 '6er',
 'phaeton',
 'x_type',
 'range_rover_sport',
 'spark',
 'viano',
 'superb',
 'spider',
 '159',
 'range_rover',
 'a8',
 'legacy',
 'amarok',
 'galant',
 's60',
 'glk',
 'wrangler',
 '9000',
 'roomster',
 'carnival',
 'i3',
 'outlander',
 'b_max',
 'antara',
 '90',
 'niva',
 'r19',
 'sirion',
 'nubira',
 'g_klasse',
 'lodgy',
 'mii',
 'crossfire',
 'range_rover_evoque',
 'terios',
 '145',
 'gl',
 'serie_2',
 'defender',
 'delta',
 'lanos',
 '200',
 's_type',
 'materia',
 'lybra',
 'croma',
 'discovery',
 'v60',
 'serie_3',
 'move',
 'kalina',
 'elefantino',
 'charade',
 'rangerover'])

    with col6:
        kilometer = st.number_input("Kilometer",min_value=0.0)



    col7, col8 = st.beta_columns(2)

    with col7:
        fule_type = st.selectbox("Fule Type",['diesel', 'petrol', 'cng', 'lpg', 'hybrid', 'electro', 'other'])

    with col8:
        brand = st.selectbox("Brand",['bmw',
 'volvo',
 'volkswagen',
 'seat',
 'mercedes_benz',
 'opel',
 'skoda',
 'toyota',
 'nissan',
 'sonstige_autos',
 'mazda',
 'mitsubishi',
 'audi',
 'fiat',
 'alfa_romeo',
 'saab',
 'peugeot',
 'land_rover',
 'subaru',
 'mini',
 'citroen',
 'jeep',
 'chrysler',
 'ford',
 'renault',
 'dacia',
 'chevrolet',
 'smart',
 'trabant',
 'suzuki',
 'hyundai',
 'honda',
 'kia',
 'jaguar',
 'daihatsu',
 'porsche',
 'rover',
 'lancia',
 'daewoo',
 'lada'])


    col9, col10 = st.beta_columns(2)

    with col9:
        non_repair_damage = st.radio("Non Repair Damage",["yes","no"])

    with col10:
        age = st.number_input("Age",min_value=0)

    if st.sidebar.button('Predict'):
        modelinfo = load_model('price_pediction.pk')
        model = modelinfo['model']
        encoders = modelinfo['encoder_dict']
    
        ve = encoders['vt'].transform(np.array([[vehchile_type]])).toarray()
        ge = encoders['gear'].transform(np.array([[gearbox]])).toarray()
        me = encoders['model'].transform(np.array([[modelname]])).toarray()
        be = encoders['brand'].transform(np.array([[brand]])).toarray()
        ne = encoders['nrd'].transform(np.array([[non_repair_damage]])).toarray()
        fe = encoders['fuel'].transform(np.array([[fule_type]])).toarray()
        # st.write('done')
        number_data = np.array([[powerPS,kilometer,age]])
        data = np.hstack((ve,ge,me,fe,be,ne,number_data))
        result = model.predict(data)
        st.success('the predicted price is:')
        st.subheader(result[0])

if st.checkbox("Visualization"):
    visualization= st.selectbox("Training Data Graphs",['Brand-Price',
    "FuleType-Price",
    "GearBox-Price",
    "Model-Price",
    "Non Repaired Damage-Price",
    "KiloMeter-Price",
    "VehicleType-Price",])


    if  visualization=="Brand-Price":
        st.image("img/brand-price.png")

    if  visualization=="FuleType-Price":
        st.image("img/fuletype-price.png")

    if  visualization=="GearBox-Price":
        st.image("img/gearbox-price.png")

    if  visualization=="Model-Price":
        st.image("img/model-price.png")

    if  visualization=="Non Repaired Damage-Price":
        st.image("img/nonrepairdamage.png")

    if  visualization=="KiloMeter-Price":
        st.image("img/price-kilometer.png")
        
    if  visualization=="VehicleType-Price":
        st.image("img/vehicletype-price.png")


