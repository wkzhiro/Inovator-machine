import streamlit as st
import requests
import os
from dotenv import load_dotenv

# .env ファイルから環境変数を読み込む
load_dotenv()

# Azure OpenAI の設定
AZURE_ENDPOINT = "https://aoai-japaneast-ab.openai.azure.com/openai/deployments/gpt-4o/chat/completions?api-version=2023-03-15-preview"
AZURE_API_KEY = os.getenv("AZURE_API_KEY", "9fe06b4aeecc43de9492752b8c2fa90f")
AZURE_MODEL = "gpt-4o"


def generate_azure_openai_response(messages):
    headers = {
        "Content-Type": "application/json",
        "api-key": AZURE_API_KEY,
    }
    payload = {
        "messages": messages,
        "model": AZURE_MODEL,
    }
    response = requests.post(AZURE_ENDPOINT, headers=headers, json=payload)
    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        raise Exception(f"Error: {response.status_code}, {response.text}")


def generate_feedback(concept, tone, perspective, additional_instructions=""):
    try:
        system_content = f"あなたは{perspective}の立場から、{tone}トーンでフィードバックを提供する専門家です。{additional_instructions}"
        messages = [
            {"role": "system", "content": system_content},
            {
                "role": "user",
                "content": f"以下の商品コンセプトについて、具体的なフィードバックを提供してください：\n\n{concept}",
            },
        ]
        return generate_azure_openai_response(messages)
    except Exception as e:
        return f"エラーが発生しました: {str(e)}"


def refine_concept(original_concept, tone, direction, additional_instructions=""):
    try:
        system_content = f"""あなたは{tone}なアプローチで、{direction}を考慮しつつコンセプトを改善する専門家です。
        元のコンセプトの核心を保ちつつ、フィードバックを基に改善案を提案してください。
        提案は元の商品カテゴリーや主要なターゲット層を変更せず、あくまで改善に焦点を当ててください。
        {additional_instructions}"""
        messages = [
            {"role": "system", "content": system_content},
            {
                "role": "user",
                "content": f"以下の元のコンセプトを参考に、改善されたコンセプトを提案してください：\n\n{original_concept}",
            },
        ]
        return generate_azure_openai_response(messages)
    except Exception as e:
        return f"エラーが発生しました: {str(e)}"


def run():
    st.title("コンセプトの壁打ち")

    # アプリのタイトル
    st.title("コンセプトフィードバック & 改善案生成アプリ")

    # --- コンセプトの入力セクション ---
    st.header("コンセプトの入力")
    concept_input = st.text_area(
        "フィードバックして欲しい商品コンセプトを入力してください",
        placeholder="例：20代向けの健康志向の新しいビール",
    )

    # --- フィードバックの追加指示 ---
    feedback_additional_instructions = st.text_area(
        "フィードバックに関する追加指示を入力してください（任意）",
        placeholder="例：不適切な飲酒について特に考慮して言及してください。フィードバックは3つの項目に分けて提供してください。",
    )

    # --- フィードバックの設定セクション ---
    st.header("フィードバックの設定")

    # フィードバックのトーン選択
    feedback_tone = st.radio(
        "フィードバックのトーンを選んでください",
        ("厳しく", "優しく", "中立的", "その他"),
    )

    custom_feedback_tone = (
        st.text_input("フィードバックのトーンを指定してください")
        if feedback_tone == "その他"
        else ""
    )

    # パーソナリティの選択
    st.subheader("フィードバックを提供するパーソナリティを選択してください")
    st.warning(
        "注意: 選択するパーソナリティの数が増えるほど、フィードバックの生成に時間がかかります。"
    )

    col1, col2 = st.columns(2)
    with col1:
        personality_colleague = st.checkbox("同僚", value=True)
        personality_marketer = st.checkbox("マーケッター")
    with col2:
        personality_consumer = st.checkbox("消費者")
        personality_custom = st.checkbox("その他")

    custom_personality = st.text_input(
        "カスタムパーソナリティを入力してください", disabled=not personality_custom
    )

    # 少なくとも1つのパーソナリティが選択されていることを確認
    if not any(
        [
            personality_colleague,
            personality_marketer,
            personality_consumer,
            personality_custom,
        ]
    ):
        st.error("少なくとも1つのパーソナリティを選択してください。")

    # --- コンセプト改善の設定セクション ---
    st.header("コンセプト改善の設定")

    # 改善案のトーン選択
    refinement_tone = st.radio(
        "改善案のトーンを選んでください",
        ("革新的", "保守的", "バランスの取れた", "その他"),
    )

    custom_refinement_tone = (
        st.text_input("改善案のトーンを指定してください")
        if refinement_tone == "その他"
        else ""
    )

    # 改善の方向性選択
    refinement_direction = st.radio(
        "改善の方向性を選んでください",
        ("市場拡大", "コスト削減", "品質向上", "その他"),
    )

    custom_refinement_direction = (
        st.text_input("改善の方向性を指定してください")
        if refinement_direction == "その他"
        else ""
    )

    # --- 改善案の追加指示 ---
    refinement_additional_instructions = st.text_area(
        "改善案に関する追加指示を入力してください（任意）",
        placeholder="例：持続可能性を重視した改善案にしてください。改善点は5つの要点にまとめて提案してください。",
    )

    # --- 生成ボタン ---
    if st.button("フィードバックと改善案を生成", type="primary"):
        if not concept_input.strip():
            st.warning("コンセプトを入力してください")
        elif not any(
            [
                personality_colleague,
                personality_marketer,
                personality_consumer,
                personality_custom,
            ]
        ):
            st.error("少なくとも1つのパーソナリティを選択してください。")
        else:
            with st.spinner("フィードバックと改善案を生成中..."):
                # フィードバックの生成
                final_tone = (
                    custom_feedback_tone if feedback_tone == "その他" else feedback_tone
                )

                feedback_perspectives = []
                if personality_colleague:
                    feedback_perspectives.append("同僚")
                if personality_marketer:
                    feedback_perspectives.append("マーケッター")
                if personality_consumer:
                    feedback_perspectives.append("消費者")
                if personality_custom and custom_personality:
                    feedback_perspectives.append(custom_personality)

                feedbacks = {}
                for perspective in feedback_perspectives:
                    feedbacks[perspective] = generate_feedback(
                        concept_input,
                        final_tone,
                        perspective,
                        feedback_additional_instructions,
                    )

                # 改善案の生成
                final_refinement_tone = (
                    custom_refinement_tone
                    if refinement_tone == "その他"
                    else refinement_tone
                )
                final_direction = (
                    custom_refinement_direction
                    if refinement_direction == "その他"
                    else refinement_direction
                )

                refined_concept = refine_concept(
                    concept_input,
                    final_refinement_tone,
                    final_direction,
                    refinement_additional_instructions,
                )

                # 結果の表示
                st.header("生成結果")

                # フィードバックの表示
                st.subheader("フィードバック")
                for perspective, feedback in feedbacks.items():
                    with st.expander(
                        f"{perspective}からのフィードバック", expanded=True
                    ):
                        st.write(feedback)

                # 改善案の表示
                with st.expander("コンセプト改善案", expanded=True):
                    st.write(refined_concept)


if __name__ == "__main__":
    run()
