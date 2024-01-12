import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputRegressor
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


df = pd.read_csv("data\cleaned_data_removedmoredata.csv")


selected_columns = ['cleaned_risk_type','cleaned_hypertension','cleaned_diabetes_mellitus','cleaned_familyhx',
                    'cleaned_exercise','cleaned_stress','cleaned_smoking','cleaned_diet','cleaned_bmi',
                    'ischemia','dyslipidemia','ejection_fraction','cleaned_peak_hr','cleaned_mets',
                    'cleaned_marital','cleaned_lives_with','cleaned_living_environment',
                    'cleaned_alcoholic','cleaned_balance','cleaned_fucntional_activity',
                    'cleaned_walking','cleaned_gait','cleaned_posture','cleaned_gender',
                    'cleaned_age','cleaned_exercise_habit_frequency','cleaned_exercise_habit_duration']

################### Risk Assessment Model #####################

selected_df = df.loc[:, selected_columns]

to_drop = ['cleaned_risk_type']
X = selected_df.drop(to_drop, axis=1)  
y = selected_df['cleaned_risk_type']  

# SelectKBest
from sklearn.feature_selection import SelectKBest, f_classif

selector = SelectKBest(score_func=f_classif, k=19)
X_kbest = selector.fit_transform(X, y)

# Assuming X_new is the result of SelectKBest and y is your target variable
X_train, X_test, y_train, y_test = train_test_split(X_kbest, y, test_size=0.2, random_state=42)

# Train a classifier (Logistic Regression) on the selected features
classifier = LogisticRegression()
classifier.fit(X_train, y_train)

####################### Recumbent Bike Model #####################
selected_columns1 = ['cleaned_risk_type','cleaned_hypertension','cleaned_diabetes_mellitus','cleaned_familyhx',
                    'cleaned_exercise','cleaned_stress','cleaned_smoking','cleaned_diet','cleaned_bmi',
                    'ischemia','dyslipidemia','ejection_fraction','cleaned_peak_hr','cleaned_mets',
                    'cleaned_marital','cleaned_lives_with','cleaned_living_environment',
                    'cleaned_alcoholic','cleaned_balance','cleaned_fucntional_activity',
                    'cleaned_walking','cleaned_gait','cleaned_posture','cleaned_gender',
                    'cleaned_age','cleaned_exercise_habit_frequency','cleaned_exercise_habit_duration',
                    'cleaned_recumbentbike_res','cleaned_recumbentbike_duration']

# Create a new DataFrame with only the selected columns
df1 = df.loc[:, selected_columns1]
targets1 = ['cleaned_recumbentbike_res', 'cleaned_recumbentbike_duration']
X1 = df1.drop(targets1, axis=1)
y1 = df1[targets1]


X_train1, X_test1, y_train1, y_test1 = train_test_split(X1, y1, test_size=0.2, random_state=42)

# Feature selection for each target variable separately
selected_features_per_target1 = []
for target_var in targets1:
    y_target1 = y1[target_var]

    # Feature selection using SelectKBest (adjust k value as needed)
    selector = SelectKBest(score_func=f_regression, k=10)
    X_selected = selector.fit_transform(X1, y_target1)

    # Save selected features indices
    selected_feature_indices = selector.get_support(indices=True)
    selected_features_per_target1.append(selected_feature_indices)

# Concatenate selected features for all target variables
all_selected_features1 = list(set().union(*selected_features_per_target1))

# Use only the selected features in your feature matrix
X_selected1 = X1.iloc[:, all_selected_features1]
selected_feature_names1 = X1.columns[all_selected_features1]

# Create a multi-output regression model (using Linear Regression as an example)
multi_output_model1 = MultiOutputRegressor(LinearRegression())

# Fit the model
multi_output_model1.fit(X_train1[X_selected1.columns], y_train1)


####################### Strengthening Exercise Model ##################
selected_columns2 = ['cleaned_risk_type','cleaned_hypertension','cleaned_diabetes_mellitus','cleaned_familyhx',
                    'cleaned_exercise','cleaned_stress','cleaned_smoking','cleaned_diet','cleaned_bmi',
                    'ischemia','dyslipidemia','ejection_fraction','cleaned_peak_hr','cleaned_mets',
                    'cleaned_marital','cleaned_lives_with','cleaned_living_environment',
                    'cleaned_alcoholic','cleaned_balance','cleaned_fucntional_activity',
                    'cleaned_walking','cleaned_gait','cleaned_posture','cleaned_gender',
                    'cleaned_age','cleaned_exercise_habit_frequency','cleaned_exercise_habit_duration',
                    'cleaned_ul_weight','cleaned_ll_weight','cleaned_strengthening_rep','cleaned_strengthening_set']

df2 = df.loc[:, selected_columns2]
targets2 = ['cleaned_ul_weight','cleaned_ll_weight','cleaned_strengthening_rep','cleaned_strengthening_set']
X2 = df2.drop(targets2, axis=1) 
y2 = df2[targets2]

X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, y2, test_size=0.2, random_state=42)

selected_features_per_target2 = []
for target_var2 in targets2:
    y_target2 = y2[target_var2]
    selector2 = SelectKBest(score_func=f_regression, k=10)
    X_selected2 = selector2.fit_transform(X2, y_target2)

    selected_feature_indices2 = selector2.get_support(indices=True)
    selected_features_per_target2.append(selected_feature_indices2)

all_selected_features2 = list(set().union(*selected_features_per_target2))

# Use only the selected features in your feature matrix
X_selected2 = X2.iloc[:, all_selected_features2]

# Print selected feature names
selected_feature_names2 = X2.columns[all_selected_features2]

multi_output_model2 = MultiOutputRegressor(LinearRegression())

# Fit the model
multi_output_model2.fit(X_train2[X_selected2.columns], y_train2)



############################################################
 
def set_state(stage):                                 
    st.session_state.stage = stage

def handle_button():
    if st.session_state.stage == 0:
        submit_button()
    if st.session_state.stage >= 1:
        risk_and_fill_data()
    if st.session_state.stage >= 2:
        prescribe_exercise()

if 'user_data' not in st.session_state:
    st.session_state.user_data = {}

def submit_button():
    col1, col2 = st.columns(2)
    with col1:
        st.session_state.user_data['hypertension'] = st.radio(
        'Do you have hypertension?',
        ['Yes','No'],
        index=None,
    )
        st.session_state.user_data['diabetes'] = st.radio(
        'Do you have diabetes?',
        ['Yes','No'],
        index = None,
    )
        st.session_state.user_data['familyhx'] = st.radio(
        'Do you have any family members with any cardiac diseases?',
        ['Yes','No'],
        index = None,
    )
    with col2:
        st.session_state.user_data['stress'] = st.radio(
        'Are you stressed?',
        ['Yes','No'],
        index = None,
    )
        st.session_state.user_data['smoking'] = st.radio(
        'Do you smoke?',
        ['Yes','No'],
        index = None,
    )
        st.session_state.user_data['diet'] = st.radio(
        'Do you control your diet?',
        ['Yes','No'],
        index = None,
    )
    
    st.session_state.user_data['dyslipidemia'] = st.radio(
        'Do you have dyslipidemia?',
        ['Yes','No'],
        index = None,
    )
    st.session_state.user_data['ef'] = st.selectbox(
        'What is your ejection fraction precentage?',
        ('Equal to or more than 50%','41% to 49%','Less than 40%'),
        index = None,
        placeholder = "Select your choice",
    )
    st.write('APMHR = Age Predicted Maximum Heart Rate')
    st.session_state.user_data['peak_hr'] = st.selectbox(
        'What is your peak heart rate precentage?',
        ('Less than 60% APMHR','60% to 80% APMHR','More than 80% APMHR'),
        index = None,
        placeholder = "Select your choice",
    )
    st.write('METs = metabolic equivalents')
    st.session_state.user_data['mets'] = st.selectbox(
        'What is your MET value?',
        ('Lower than 3','3 to 6','Higher than 6'),
        index = None,
        placeholder = "Select your choice",
    )
    st.session_state.user_data['marital'] = st.selectbox(
        'What is your marital status?',
        ('Single','Divorced','Married'),
        index = None,
        placeholder = "Select your choice",
    )
    st.session_state.user_data['alcoholic'] = st.selectbox(
        'Do you drink?',
        ('No','Occasional drinker','Social drinker'),
        index = None,
        placeholder = "Select your choice",
    )
    st.session_state.user_data['functional_activity'] = st.selectbox(
        'Are you dependant or independant to perform daily activites?',
        ('Dependant','Independant'),
        index = None,
        placeholder = "Select your choice",
    )
    st.session_state.user_data['walking'] = st.radio(
        'Are you able to walk by your own?',
        ('Yes','No'),
        index = None,
    )
    st.write('Gait = The pattern you walk')
    st.session_state.user_data['gait'] = st.selectbox(
        'Normal or abnormal gait?',
        ('Normal','Abnormal'),
        index = None,
        placeholder = "Select your choice",
    )
    st.session_state.user_data['posture'] = st.selectbox(
        'Normal or abnormal posture?',
        ('Normal','Abnormal'),
        index = None,
        placeholder = "Select your choice",
    )
    st.session_state.user_data['gender'] = st.selectbox(
        'Gender',
        ('Male','Female'),
        index = None,
        placeholder = "Select your choice",
    )
    st.session_state.user_data['age'] = st.selectbox(
        'How old are you?',
        ('Younger than 40','40 - 50 years old','60 - 79 years old','Elder than 80 years old'),
        index = None,
        placeholder = "Select your choice",
    )
    st.session_state.user_data['exercise_habit_duration'] = st.selectbox(
        'How long do you usually exercise per session?',
        ('Less than 10 mins','10 - 19 mins','20 - 29 mins',
         '30 - 39 mins','30 - 39 mins','40 - 49 mins',
         '50 - 59 mins',' More than 60 mins'),
        index = None,
        placeholder = "Select your choice",
    )
    st.button("Submit", on_click=set_state, args=[1])
    
def risk_and_fill_data():
    user_data = st.session_state.user_data
    user_data2 = pd.DataFrame({
            'hypertension': [1 if user_data.get('hypertension') == 'Yes' else 0],
            'diabetes': [1 if user_data.get('diabetes') == 'Yes' else 0],
            'familyhx': [1 if user_data.get('familyhx') == 'Yes' else 0],
            'stress':[1 if user_data.get('stress') == 'Yes' else 0],
            'smoking': [1 if user_data.get('smoking') == 'Yes' else 0],
            'diet': [1 if user_data.get('diet') == 'Yes' else 0],
            'dyslipiedmia': [1 if user_data.get('dyslipidemia') == 'Yes' else 0],
            'ef': [0 if user_data.get('ef') == 'Equal to or more than 50%' else (1 if user_data.get('ef') == '41% to 49%' else 2)],
            'peak_hr': [0 if user_data.get('peak_hr') == 'Less than 60% APMHR' else (1 if user_data.get('peak_hr') == '60% to 80% APMHR' else 2)],
            'mets': [0 if user_data.get('mets') == 'Lower than 3' else (1 if user_data.get('mets') == '3 to 6' else 2)],
            'marital':[0 if user_data.get('marital') == 'Single' else (1 if user_data.get('marital') == 'Divorced' else 2)],
            'alcoholic': [0 if user_data.get('alcoholic') == 'No' else (1 if user_data.get('alcoholic') == 'Occasional drinker' else 2)],
            'functional_activity': [1 if user_data.get('functional_activity') == 'Independant' else 0],
            'walking': [1 if user_data.get('walking') == 'Yes' else 0],
            'gait': [1 if user_data.get('gait') == 'Normal' else 0],
            'posture': [1 if user_data.get('posture') == 'Normal' else 0],
            'gender': [1 if user_data.get('gender') == 'Female' else 0],
            'age': [0 if user_data.get('age') == 'Younger than 40' else (1 if user_data.get('age') == '40 - 50 years old' else 
                                                        2 if user_data.get('age') == '60 - 79 years old' 
                                                        else 3)],
            'exercise_habit_duration': [0 if user_data.get('exercise_habit_duration') == 'Less than 10 mins' else 
                                        (1 if user_data.get('exercise_habit_duration') == '10 - 19 mins' else
                                        2 if user_data.get('exercise_habit_duration') == '20 - 29 mins' else
                                        3 if user_data.get('exercise_habit_duration') == '30 - 39 mins' else
                                        4 if user_data.get('exercise_habit_duration') == '40 - 49 mins' else
                                        5 if user_data.get('exercise_habit_duration') == '50 - 59 mins' 
                                        else 6)],
        
            })

    predicted_class = classifier.predict(pd.DataFrame(user_data2))
    predicted_label = 'Low Risk' if predicted_class == 0 else 'Moderate Risk'

    st.write(f'Predicted Risk Level: {predicted_label}')

    if predicted_label == 'Moderate Risk':
        st.write('Target heart rate = Maximum heart rate x 50%')
    else:
        st.write('Target heart rate = Maximum heart rate x 60%')
    
    st.session_state.user_data['risk'] = predicted_label


    ################ FOR RECUMBENT EXERCISE PRESCRIPTION ##################
        
    st.write('Fill in to prescribe recumbent bike exercise')
    
    st.session_state.user_data['bmi'] = st.selectbox(
        "BMI",
        ("Underweight","Normal","Overweight","Obese"),
        placeholder = "Select your BMI",
        index = None,
        )
    st.session_state.user_data['lives_with'] = st.selectbox(
        "Lives with",
        ("Alone","Family","Friends"),
        placeholder = "Select your choice",
        index = None,
        )
    st.session_state.user_data['balance'] = st.selectbox(
        "Balance in Sitting and Standing",
        ("Abnormal","Normal"),
        placeholder = "Select your choice",
        index = None,
        )
    st.session_state.user_data['exercise_habit_frequency'] = st.selectbox(
        "Exercise habit frequency (times/week)",
        ("0","1","2","3","4","5","6","7"),
        placeholder = "Select your choice",
        index = None,
        )
                
    #for strengthening exercise
    st.session_state.user_data['exercise'] = st.radio(
        "Do you exercise regularly?",
        ("Yes","No"),
        index = None,
        )
                                    
    st.button("Prescribe", on_click=set_state, args=[2])

def prescribe_exercise():
    user_data = st.session_state.user_data
    predicted_label = st.session_state.user_data.get('predicted_label', 'Not available')
    user_data_recumbent = pd.DataFrame({
        'risk_level': [1 if predicted_label == "Moderate" else 0],
        'hypertension': [1 if user_data.get('hypertension') == 'Yes' else 0],
        'diabetes': [1 if user_data.get('diabetes') == 'Yes' else 0],
        'diet': [1 if user_data.get('diet') == 'Yes' else 0],
        'bmi':[0 if user_data.get('bmi') == "Underweight" else 
               (1 if user_data.get('bmi') == "Normal" else 
                2 if user_data.get('bmi') == "Overweight" else
                3)],
        'dyslipiedmia': [1 if user_data.get('dyslipidemia') == 'Yes' else 0],       
        'peak_hr': [0 if user_data.get('peak_hr') == 'Less than 60% APMHR' else 
                    (1 if user_data.get('peak_hr') == '60% to 80% APMHR' else 2)],
        'lives_with': [0 if user_data.get('lives_with') == "Alone" else 
                        (1 if user_data.get('lives_with') == "Family" else 
                        2)], 
        'balance': [1 if  user_data.get('balance') == 'Normal' else 0], 
        'functional_activity': [1 if user_data.get('functional_activity') == 'Independant' else 0],                 
        'walking': [1 if user_data.get('walking') == 'Yes' else 0],
        'gait': [1 if user_data.get('gait') == 'Normal' else 0],
        'posture': [1 if user_data.get('posture') == 'Normal' else 0],
        'gender': [1 if user_data.get('gender') == 'Female' else 0],
        'age': [0 if user_data.get('age') == 'Younger than 40' else 
                (1 if user_data.get('age') == '40 - 50 years old' else 
                2 if user_data.get('age') == '60 - 79 years old' 
                else 3)],
        'exercise_habit_frequency': [0 if user_data.get('exercise_habit_frequency') == '0' else 
                                    (1 if user_data.get('exercise_habit_frequency') == '1' else
                                    2 if user_data.get('exercise_habit_frequency') == '2' else
                                    3 if user_data.get('exercise_habit_frequency') == '3' else
                                    4 if user_data.get('exercise_habit_frequency') == '4' else
                                    5 if user_data.get('exercise_habit_frequency') == '5' else
                                    6 if user_data.get('exercise_habit_frequency') == '6' 
                                    else 7)],        
        })

    st.write("Exercise")
    #st.write(user_data_recumbent)
    sample_input1 = user_data_recumbent.values.tolist()
    #st.write(sample_input1)
    predicted_outputs1 = multi_output_model1.predict(pd.DataFrame(sample_input1, columns=selected_feature_names1))
    for target_var, predicted_value in zip(targets1, predicted_outputs1[0]):
        st.write(f"{target_var}: {int(predicted_value)}")

    ###################### STRENGTHENING EXERCISE #####################
        
    user_data_strengthening = pd.DataFrame({
        'risk_level': [1 if predicted_label == "Moderate" else 0],
        'hypertension': [1 if user_data.get('hypertension') == 'Yes' else 0],
        'diabetes': [1 if user_data.get('diabetes') == 'Yes' else 0],
        'exercise':[1 if user_data.get('exercise') == 'Yes' else 0],
        'stress':[1 if user_data.get('stress') == 'Yes' else 0],
        'smoking': [1 if user_data.get('smoking') == 'Yes' else 0],
        'diet': [1 if user_data.get('diet') == 'Yes' else 0],
        'bmi':[0 if user_data.get('bmi') == "Underweight" else 
               (1 if user_data.get('bmi') == "Normal" else 
                2 if user_data.get('bmi') == "Overweight" else
                3)],
        'dyslipiedmia': [1 if user_data.get('dyslipidemia') == 'Yes' else 0],       
        'peak_hr': [0 if user_data.get('peak_hr') == 'Less than 60% APMHR' else 
                    (1 if user_data.get('peak_hr') == '60% to 80% APMHR' else 2)],
        
        'mets': [0 if user_data.get('mets') == 'Lower than 3' else (1 if user_data.get('mets') == '3 to 6' else 2)],
        'lives_with': [0 if user_data.get('lives_with') == "Alone" else 
                        (1 if user_data.get('lives_with') == "Family" else 
                        2)], 
        'alcoholic': [0 if user_data.get('alcoholic') == 'No' else (1 if user_data.get('alcoholic') == 'Occasional drinker' else 2)],
        'balance': [1 if  user_data.get('balance') == 'Normal' else 0], 
        'functional_activity': [1 if user_data.get('functional_activity') == 'Independant' else 0],                 
        'gait': [1 if user_data.get('gait') == 'Normal' else 0],
        'gender': [1 if user_data.get('gender') == 'Female' else 0],
        'age': [0 if user_data.get('age') == 'Younger than 40' else 
                (1 if user_data.get('age') == '40 - 50 years old' else 
                2 if user_data.get('age') == '60 - 79 years old' 
                else 3)],
        'exercise_habit_frequency': [0 if user_data.get('exercise_habit_frequency') == '0' else 
                                    (1 if user_data.get('exercise_habit_frequency') == '1' else
                                    2 if user_data.get('exercise_habit_frequency') == '2' else
                                    3 if user_data.get('exercise_habit_frequency') == '3' else
                                    4 if user_data.get('exercise_habit_frequency') == '4' else
                                    5 if user_data.get('exercise_habit_frequency') == '5' else
                                    6 if user_data.get('exercise_habit_frequency') == '6' 
                                    else 7)],
        'exercise_habit_duration': [0 if user_data.get('exercise_habit_duration') == 'Less than 10 mins' else 
                                        (1 if user_data.get('exercise_habit_duration') == '10 - 19 mins' else
                                        2 if user_data.get('exercise_habit_duration') == '20 - 29 mins' else
                                        3 if user_data.get('exercise_habit_duration') == '30 - 39 mins' else
                                        4 if user_data.get('exercise_habit_duration') == '40 - 49 mins' else
                                        5 if user_data.get('exercise_habit_duration') == '50 - 59 mins' 
                                        else 6)],             
                        
        })
    
    sample_input2 = user_data_strengthening.values.tolist()
    predicted_outputs2 = multi_output_model2.predict(pd.DataFrame(sample_input2, columns=selected_feature_names2))

    for target_var, predicted_value in zip(targets2, predicted_outputs2[0]):
        st.write(f"{target_var}: {int(predicted_value)}")




if __name__ == '__main__':
    if 'stage' not in st.session_state:
        st.session_state.stage = 0

    handle_button()