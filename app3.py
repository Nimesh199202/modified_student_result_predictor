import streamlit as st
import pandas as pd
import joblib
import pickle
import os

# -----------------------------
# Safe loader for models
# -----------------------------
def safe_load_model(path):
    if not os.path.exists(path):
        st.error(f"Model file not found: {path}")
        return None
    try:
        return joblib.load(path)
    except Exception as e:
        st.warning(f"joblib.load failed for {path}: {e}. Trying pickle with latin1 encoding...")
        try:
            with open(path, "rb") as f:
                return pickle.load(f, encoding="latin1")
        except Exception as e2:
            st.error(f"Failed to load model {path}: {e2}")
            return None

# Load models & scaler
math_model = safe_load_model("gradient_boost_mathes_finalized.pkl")
science_model = safe_load_model("gbr_science_model.pkl")
science_scaler = safe_load_model("gbr_science_scaler.pkl")

# -----------------------------
# Mappings
# -----------------------------
family_income_mapping = {1: "Below 25,000", 2: "25,000 - 50,000", 3: "50,000 - 75,000", 4: "75,000 - 100,000", 5: "Above 100,000"}
edu_mapping = {0: "Don't go to school", 1: "Grade (1-5)", 2: "Grade (6-11)", 3: "Pass (O/L)", 4: "Diploma", 5: "Pass (A/L)", 6: "Degree holder"}
relationship_mapping = {0: "Very bad / Bad", 1: "Average", 2: "Good", 3: "Very good"}
food_mapping = {1: "Very sweet / Very salty / Junk food", 2: "Healthy food"}
internet_mapping = {2: "Have", 1: "No"}
nursery_mapping = {1: "Yes", 0: "No"}
higheredu_mapping = {1: "Yes", 0: "No"}
transport_mapping = {1: "Personal vehicle", 0: "Other"}
extraclasses_mapping = {2: "Yes", 1: "No"}
gender_mapping = {1: "Female", 0: "Male"}

def rev_map(d): return {v: k for k, v in d.items()}
family_income_rev = rev_map(family_income_mapping)
mother_edu_rev = rev_map(edu_mapping)
father_edu_rev = rev_map(edu_mapping)
relationship_rev = rev_map(relationship_mapping)
food_rev = rev_map(food_mapping)
internet_rev = rev_map(internet_mapping)
nursery_rev = rev_map(nursery_mapping)
higheredu_rev = rev_map(higheredu_mapping)
transport_rev = rev_map(transport_mapping)
extraclasses_rev = rev_map(extraclasses_mapping)
gender_rev = rev_map(gender_mapping)

# -----------------------------
# Features
# -----------------------------
math_features = [
    'පළවන වාරයේ ගණිතය ලකුණු',
    'දෙවන වාරයේ ගණිතය ලකුණු ',
    '6. මවගේ \u200d ඉහලම අධ්\u200dයාපන සුදුසුකම?',
    '13.ඔබ අමතර පන්ති වලට සහභාගී වෙන වද?',
    '5.පියාගේ ඉහලම  අධ්\u200dයාපන සුදුසුකම?',
    '3.සාමාන්\u200dය පවුලේ ආදායම',
    '27.ඔබ නිතර ගන්නා ආහාර...?',
    '8. දිනකට  නිවසේදී පාඩම් කිරීම සඳහා වැය කරන පැය සංඛ්\u200dයාව',
    '22. පාසලට   ඔබ පැමිණෙන්නේ කෙසේද?_පුද්ගලික වාහන භාවිතයෙන්(බයිසිකලය,මෝටර් සයිකලය,ත්\u200dරීරෝද රථය,වෑන් රථයක........)',
    '16.පවුලේ සබඳතා තත්ත්වය?',
    '15.ඔබ  පෙර පාසල් ගොස් තිබේද?',
    '14.ගෙදර අන්තර්ජාල පහසුකම් තිබේද?',
    '2.ස්ත්\u200dරී පුරුෂ භාවය_ස්ත්\u200dරී',
    '17.ඔබ උසස් අධ්\u200dයාපනය ලබා ගැනීමට කැමතිද?'
]

science_features = [
    "පළවන වාරයේ විද්\u200dයා ලකුණු",
    "දෙවන වාරයේ විද්\u200dයා ලකුණු",
    "6. මවගේ \u200d ඉහලම අධ්\u200dයාපන සුදුසුකම?",
    "13.ඔබ අමතර පන්ති වලට සහභාගී වෙන වද?",
    "27.ඔබ නිතර ගන්නා ආහාර...?",
    "22. පාසලට   ඔබ පැමිණෙන්නේ කෙසේද?_පුද්ගලික වාහන භාවිතයෙන්(බයිසිකලය,මෝටර් සයිකලය,ත්\u200dරීරෝද රථය,වෑන් රථයක........)",
    "3.සාමාන්\u200dය පවුලේ ආදායම",
    "5.පියාගේ ඉහලම  අධ්\u200dයාපන සුදුසුකම?",
    "16.පවුලේ සබඳතා තත්ත්වය?",
    "2.ස්ත්\u200dරී පුරුෂ භාවය_ස්ත්\u200dරී"
]

# -----------------------------
# Streamlit UI Styling
# -----------------------------
st.set_page_config(page_title="Student Marks Predictor", layout="centered", page_icon="🎓")
st.markdown("""
<style>
body {background: linear-gradient(to bottom right, #e6f0ff, #ffffff);}
h1, h2, h3 {color: #1B4F72;}
.stButton>button {background-color:#1B4F72;color:white;font-weight:bold;height:3em;width:100%;}
</style>
""", unsafe_allow_html=True)

# Logo
st.image("logo.png", width=120)  # replace with your logo path
st.title("🎓 Student Marks Predictor")
st.write("Predict 3rd-term Maths & Science marks easily — Sinhala labels & user-friendly inputs.")

# Tabs for Maths & Science
tab1, tab2 = st.tabs(["📘 Maths Prediction", "🔬 Science Prediction"])

# -----------------------------
# Maths Tab
# -----------------------------
with tab1:
    st.header("Maths Inputs")
    with st.form("math_form"):
        math_term1 = st.number_input("1st Term Maths", min_value=0.0, max_value=200.0)
        math_term2 = st.number_input("2nd Term Maths", min_value=0.0, max_value=200.0)
        study_hours = st.number_input("Study hours/day", min_value=0.0, max_value=24.0, value=1.0, step=0.5)
        st.markdown("---")
        st.subheader("Background Information")
        family_income_label = st.selectbox("Family income", list(family_income_mapping.values()))
        mother_edu_label = st.selectbox("Mother's education", list(edu_mapping.values()))
        father_edu_label = st.selectbox("Father's education", list(edu_mapping.values()))
        extra_classes_label = st.selectbox("Extra classes", list(extraclasses_mapping.values()))
        food_label = st.selectbox("Food habits", list(food_mapping.values()))
        transport_label = st.selectbox("Transport mode", list(transport_mapping.values()))
        relationship_label = st.selectbox("Family relationship", list(relationship_mapping.values()))
        nursery_label = st.selectbox("Nursery school", list(nursery_mapping.values()))
        internet_label = st.selectbox("Internet at home", list(internet_mapping.values()))
        gender_label = st.selectbox("Gender", list(gender_mapping.values()))
        higheredu_label = st.selectbox("Interest in higher education", list(higheredu_mapping.values()))
        submit_math = st.form_submit_button("Predict Maths")
    
    if submit_math:
        try:
            math_input = pd.DataFrame([{
                'පළවන වාරයේ ගණිතය ලකුණු': math_term1,
                'දෙවන වාරයේ ගණිතය ලකුණු ': math_term2,
                '6. මවගේ \u200d ඉහලම අධ්\u200dයාපන සුදුසුකම?': mother_edu_rev[mother_edu_label],
                '13.ඔබ අමතර පන්ති වලට සහභාගී වෙන වද?': extraclasses_rev[extra_classes_label],
                '5.පියාගේ ඉහලම  අධ්\u200dයාපන සුදුසුකම?': father_edu_rev[father_edu_label],
                '3.සාමාන්‍ය පවුලේ ආදායම': family_income_rev[family_income_label],
                '27.ඔබ නිතර ගන්නා ආහාර...?': food_rev[food_label],
                '8. දිනකට  නිවසේදී පාඩම් කිරීම සඳහා වැය කරන පැය සංඛ්\u200dයාව': study_hours,
                '22. පාසලට   ඔබ පැමිණෙන්නේ කෙසේද?_පුද්ගලික වාහන භාවිතයෙන්(බයිසිකලය,මෝටර් සයිකලය,ත්\u200dරීරෝද රථය,වෑන් රථයක........)': transport_rev[transport_label],
                '16.පවුලේ සබඳතා තත්ත්වය?': relationship_rev[relationship_label],
                '15.ඔබ  පෙර පාසල් ගොස් තිබේද?': nursery_rev[nursery_label],
                '14.ගෙදර අන්තර්ජාල පහසුකම් තිබේද?': internet_rev[internet_label],
                '2.ස්ත්‍රී පුරුෂ භාවය_ස්ත්\u200dරී': gender_rev[gender_label],
                '17.ඔබ උසස් අධ්\u200dයාපනය ලබා ගැනීමට කැමතිද?': higheredu_rev[higheredu_label]
            }])
            math_input = math_input[math_features]
            math_pred = math_model.predict(math_input)[0]
            st.markdown(f"<div style='background-color:#d6eaf8;padding:15px;border-radius:10px;text-align:center;font-size:24px;'>📘 Predicted Maths Marks: {math_pred:.2f}</div>", unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Math prediction failed: {e}")

# -----------------------------
# Science Tab
# -----------------------------
with tab2:
    st.header("Science Inputs")
    with st.form("science_form"):
        science_term1 = st.number_input("1st Term Science", min_value=0.0, max_value=200.0)
        science_term2 = st.number_input("2nd Term Science", min_value=0.0, max_value=200.0)
        st.markdown("---")
        st.subheader("Background Information")
        mother_edu_label_sci = st.selectbox("Mother's education", list(edu_mapping.values()))
        father_edu_label_sci = st.selectbox("Father's education", list(edu_mapping.values()))
        extra_classes_label_sci = st.selectbox("Extra classes", list(extraclasses_mapping.values()))
        food_label_sci = st.selectbox("Food habits", list(food_mapping.values()))
        transport_label_sci = st.selectbox("Transport mode", list(transport_mapping.values()))
        relationship_label_sci = st.selectbox("Family relationship", list(relationship_mapping.values()))
        family_income_label_sci = st.selectbox("Family income", list(family_income_mapping.values()))
        gender_label_sci = st.selectbox("Gender", list(gender_mapping.values()))
        submit_sci = st.form_submit_button("Predict Science")
    
    if submit_sci:
        try:
            sci_input = pd.DataFrame([{
                "පළවන වාරයේ විද්\u200dයා ලකුණු": science_term1,
                "දෙවන වාරයේ විද්\u200dයා ලකුණු": science_term2,
                "6. මවගේ \u200d ඉහලම අධ්\u200dයාපන සුදුසුකම?": mother_edu_rev[mother_edu_label_sci],
                "13.ඔබ අමතර පන්ති වලට සහභාගී වෙන වද?": extraclasses_rev[extra_classes_label_sci],
                "27.ඔබ නිතර ගන්නා ආහාර...?": food_rev[food_label_sci],
                "22. පාසලට   ඔබ පැමිණෙන්නේ කෙසේද?_පුද්ගලික වාහන භාවිතයෙන්(බයිසිකලය,මෝටර් සයිකලය,ත්\u200dරීරෝද රථය,වෑන් රථයක........)": transport_rev[transport_label_sci],
                "3.සාමාන්‍ය පවුලේ ආදායම": family_income_rev[family_income_label_sci],
                "5.පියාගේ ඉහලම  අධ්\u200dයාපන සුදුසුකම?": father_edu_rev[father_edu_label_sci],
                "16.පවුලේ සබඳතා තත්ත්වය?": relationship_rev[relationship_label_sci],
                "2.ස්ත්\u200dරී පුරුෂ භාවය_ස්ත්\u200dරී": gender_rev[gender_label_sci]
            }])
            sci_input = sci_input[science_features]
            sci_input_scaled = science_scaler.transform(sci_input) if science_scaler else sci_input
            sci_pred = science_model.predict(sci_input_scaled)[0]
            st.markdown(f"<div style='background-color:#d4efdf;padding:15px;border-radius:10px;text-align:center;font-size:24px;'>🔬 Predicted Science Marks: {sci_pred:.2f}</div>", unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Science prediction failed: {e}")
