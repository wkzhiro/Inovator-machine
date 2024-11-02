import streamlit as st
import apps.concept_app as concept_app
import apps.design_app as design_app
import apps.BeerStorm2_app as idea_app


# サイドバーでアプリを選択
st.sidebar.title("アプリの選択")
app_mode = st.sidebar.radio(
    "アプリを選んでください", ["コンセプトの壁打ち", "デザイン案の生成","アイディアの生成"]
)

# アプリを選択して表示
if app_mode == "コンセプトの壁打ち":
    concept_app.run()  # concept_app.pyのメイン関数を呼び出す
elif app_mode == "デザイン案の生成":
    design_app.run()  # design_app.pyのメイン関数を呼び出す
elif app_mode == "アイディアの生成":
    idea_app.run()  # design_app.pyのメイン関数を呼び出す
