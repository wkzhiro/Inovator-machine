import streamlit as st
import requests
from dotenv import load_dotenv
import os
from io import BytesIO
import time
from openai import AzureOpenAI

# .envファイルから環境変数を読み込む
load_dotenv()

# Azure OpenAI の設定
AZURE_CHAT_ENDPOINT = os.getenv("AZURE_CHAT_ENDPOINT")
AZURE_DALLE_ENDPOINT = os.getenv("AZURE_DALLE_ENDPOINT")
AZURE_API_KEY = os.getenv("AZURE_API_KEY")
DALLE_API_KEY = os.getenv("DALLE_API_KEY")
if not AZURE_API_KEY:
    st.error("AZURE_API_KEY が設定されていません。.envファイルを確認してください。")
    st.stop()

chat_client = AzureOpenAI(
    api_key = AZURE_API_KEY,
    api_version = "2024-02-01",
    azure_endpoint = AZURE_CHAT_ENDPOINT
)

# dalle_client = AzureOpenAI(
#     api_key = DALLE_API_KEY,
#     api_version = "2024-02-01",
#     azure_endpoint = AZURE_CHAT_ENDPOINT
# )
# OpenAI API の設定（テキスト生成用）
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# if not OPENAI_API_KEY:
#     st.error("OPENAI_API_KEY が設定されていません。.envファイルを確認してください。")
#     st.stop()

# import openai

# openai.api_key = OPENAI_API_KEY


def create_image_prompt(
    design_concept, container_type, container_size, additional_instructions
):
    
    prompt = f"""
    ビール{container_type}のパッケージデザイン。
    サイズ: {container_size}
    コンセプト: {design_concept}
    追加指示: {additional_instructions if additional_instructions else 'なし'}
    デザインは{container_type}の形状に合わせて最適化されています。
    写実的なスタイルで、製品として実在しそうなデザインにしてください。
    """
    return prompt


def generate_image_with_dalle(prompt):
    headers = {
        "Content-Type": "application/json",
        "api-key": DALLE_API_KEY,
    }
    payload = {
        "prompt": prompt,
        "n": 1,
        "size": "1024x1024",
    }

    
    response = requests.post(AZURE_DALLE_ENDPOINT, headers=headers, json=payload)
    if response.status_code == 200:
        return response.json()["data"][0]["url"]
    elif response.status_code == 429:
        st.warning("API利用制限に達しました。少し待ってから再試行します。")
        time.sleep(10)  # 10秒待機
        return None
    else:
        st.error(
            f"画像生成中にエラーが発生しました: {response.status_code}, {response.text}"
        )
        return None


def get_design_suggestion(
    design_concept, container_type, container_size, additional_instructions
):
    
    try:
        response = chat_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": "あなたはビールのパッケージデザインの専門家です。与えられた条件に基づいて、具体的なデザイン案を提案してください。",
                },
                {
                    "role": "user",
                    "content": f"""
                    以下の条件でビールのパッケージデザインを提案してください：
                    - 容器タイプ: {container_type}
                    - サイズ: {container_size}
                    - デザインコンセプト: {design_concept}
                    - 追加指示: {additional_instructions if additional_instructions else 'なし'}
                    """,
                },
            ],
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"デザイン案生成中にエラーが発生しました: {str(e)}")
        return None


def get_image_data(url):
    response = requests.get(url)
    return BytesIO(response.content)


def run():
    if "generated_designs" not in st.session_state:
        st.session_state.generated_designs = []

    st.title("ビールパッケージデザイン生成器 (DALL-E 3版)")

    st.header("デザインコンセプト入力")
    design_concept = st.text_area(
        "先に作成した商品コンセプトを入力するなど、デザインしたいコンセプトを入力してください。",
        placeholder="例：若者向けの爽やかな夏限定デザイン",
    )

    col1, col2 = st.columns(2)
    with col1:
        container_type = st.radio("容器タイプ", ("缶", "瓶"))
    with col2:
        if container_type == "缶":
            container_size = st.radio("サイズ", ("350ml", "500ml"))
        else:
            container_size = st.radio("サイズ", ("小瓶", "中瓶", "大瓶"))

    additional_instructions = st.text_area(
        "追加のデザイン指示（任意）",
        placeholder="例：金色のロゴを使用、和風の要素を取り入れる、など具体的な指示を入力してください",
    )

    if st.button("デザインを生成"):
        if not design_concept.strip():
            st.warning("デザインコンセプトを入力してください")
        else:
            with st.spinner("デザイン案を生成中..."):
                design_suggestion = get_design_suggestion(
                    design_concept,
                    container_type,
                    container_size,
                    additional_instructions,
                )
                if design_suggestion:
                    st.subheader("生成されたデザイン案")
                    st.write(design_suggestion)

                    with st.spinner("デザインイメージを生成中..."):
                        image_prompt = create_image_prompt(
                            design_concept,
                            container_type,
                            container_size,
                            additional_instructions,
                        )
                        with st.expander("生成された画像プロンプト"):
                            st.write(image_prompt)

                        image_urls = []
                        for i in range(3):  # 3つのデザインを生成
                            image_url = generate_image_with_dalle(image_prompt)
                            if image_url:
                                image_urls.append(image_url)
                            else:
                                st.warning(
                                    f"デザイン {i+1} の生成に失敗しました。スキップします。"
                                )
                            time.sleep(2)  # APIレート制限を回避するために2秒待機

                        if image_urls:
                            for i, url in enumerate(image_urls):
                                st.subheader(f"デザイン案 {i+1}")
                                st.image(
                                    url,
                                    caption=f"生成されたデザインイメージ {i+1}",
                                    use_column_width=True,
                                )

                                # ダウンロードボタンの追加
                                image_data = get_image_data(url)
                                st.download_button(
                                    label=f"デザイン {i+1} をダウンロード",
                                    data=image_data,
                                    file_name=f"generated_design_{i+1}.png",
                                    mime="image/png",
                                )

                            st.session_state.generated_designs.append(
                                {
                                    "concept": design_concept,
                                    "suggestion": design_suggestion,
                                    "image_urls": image_urls,
                                }
                            )
                            if len(st.session_state.generated_designs) > 10:
                                st.session_state.generated_designs.pop(0)
                        else:
                            st.error("すべての画像生成に失敗しました。")

    if st.session_state.generated_designs:
        st.header("過去の生成結果")
        for i, design in enumerate(reversed(st.session_state.generated_designs)):
            with st.expander(
                f"デザイン案セット {len(st.session_state.generated_designs)-i}: {design['concept'][:30]}..."
            ):
                st.write(design["suggestion"])
                for j, url in enumerate(design["image_urls"]):
                    st.subheader(f"デザイン {j+1}")
                    st.image(
                        url,
                        caption=f"デザイン {len(st.session_state.generated_designs)-i}, 画像 {j+1}",
                        use_column_width=True,
                    )
                    image_data = get_image_data(url)
                    st.download_button(
                        label=f"デザイン {len(st.session_state.generated_designs)-i}, 画像 {j+1} をダウンロード",
                        data=image_data,
                        file_name=f"design_{len(st.session_state.generated_designs)-i}_{j+1}.png",
                        mime="image/png",
                    )


if __name__ == "__main__":
    run()
