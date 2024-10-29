import streamlit as st
from dataclasses import dataclass  # 追加
import pandas as pd
import requests
import datetime
import uuid
import logging
import time
import os
from dotenv import load_dotenv
from google.oauth2.service_account import Credentials
import gspread
from typing import Dict, List, Optional, Union, Any
from enum import Enum

# 必要な追加パッケージのインストール
import json
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# ロギングの設定
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ニードスコープの種類を定義するクラス
class NeedscopeType(Enum):
    STABILITY = "安定志向"
    HARMONY = "協調志向"
    BELONGING = "同調志向"
    INDEPENDENCE = "独立志向"
    POWER = "支配志向"
    FUN = "享楽志向"


# API設定を保持するクラス
@dataclass
class APIConfig:
    azure_endpoint: str
    azure_api_key: str
    azure_model: str
    openai_api_key: str
    google_sheets_id: str
    google_credentials: dict


# ペルソナ情報を保持するクラス
@dataclass
class Persona:
    persona_id: str
    age: int
    gender: str
    occupation: str
    lifestyle: str
    drinking_habits: str
    beer_preferences: str


# アイデア情報を保持するクラス
@dataclass
class Idea:
    idea_id: str
    session_id: str
    persona_id: str
    needscope_type: NeedscopeType
    concept_name: str
    description: str
    key_features: List[str]
    target_price: str
    value_proposition: str
    evaluation_score: float
    created_at: datetime.datetime


# BeerStormSystem クラス
class BeerStormSystem:
    def __init__(self):
        """システムの初期化"""
        self.api_config = self.setup_apis()
        self.setup_session_state()

    def setup_apis(self) -> APIConfig:
        """APIの設定を行い、設定オブジェクトを返す"""
        try:
            load_dotenv()
            required_vars = {
                "AZURE_ENDPOINT": os.getenv("AZURE_ENDPOINT"),
                "AZURE_API_KEY": os.getenv("AZURE_API_KEY"),
                "AZURE_MODEL": os.getenv("AZURE_MODEL"),
                "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
                "GOOGLE_SHEETS_ID": os.getenv("GOOGLE_SHEETS_ID"),
                "GOOGLE_CREDENTIALS": os.getenv("GOOGLE_CREDENTIALS"),
            }

            missing_vars = [k for k, v in required_vars.items() if not v]
            if missing_vars:
                raise ValueError(
                    f"必要な環境変数が設定されていません: {', '.join(missing_vars)}"
                )

            google_credentials = json.loads(required_vars["GOOGLE_CREDENTIALS"])

            api_config = APIConfig(
                azure_endpoint=required_vars["AZURE_ENDPOINT"],
                azure_api_key=required_vars["AZURE_API_KEY"],
                azure_model=required_vars["AZURE_MODEL"],
                openai_api_key=required_vars["OPENAI_API_KEY"],
                google_sheets_id=required_vars["GOOGLE_SHEETS_ID"],
                google_credentials=google_credentials,
            )

            logger.info("API設定の読み込みが完了しました")
            return api_config

        except Exception as e:
            logger.error(f"API設定エラー: {str(e)}")
            st.error("API設定の初期化に失敗しました。環境変数を確認してください。")
            raise

    def generate_ideas(
        self, persona: Persona, needscope_type: NeedscopeType, batch_size: int
    ) -> List[Idea]:
        """アイディアの生成"""
        try:
            ideas = []
            # 仮のニードスコープデータ
            NEEDSCOPE_DATA = {
                NeedscopeType.STABILITY: {
                    "name": "安定志向",
                    "description": "安定を重視",
                    "keywords": ["安定", "持続"],
                },
                NeedscopeType.HARMONY: {
                    "name": "協調志向",
                    "description": "協調を重視",
                    "keywords": ["協調", "共感"],
                },
                NeedscopeType.BELONGING: {
                    "name": "同調志向",
                    "description": "グループに溶け込む",
                    "keywords": ["グループ", "連帯"],
                },
                NeedscopeType.INDEPENDENCE: {
                    "name": "独立志向",
                    "description": "個人の自由を尊重",
                    "keywords": ["自由", "冒険"],
                },
                NeedscopeType.POWER: {
                    "name": "支配志向",
                    "description": "リーダーシップを発揮",
                    "keywords": ["リーダー", "支配"],
                },
                NeedscopeType.FUN: {
                    "name": "享楽志向",
                    "description": "楽しさを追求",
                    "keywords": ["楽しみ", "遊び"],
                },
            }
            needscope_info = NEEDSCOPE_DATA[needscope_type]

            prompt = f"""
            以下の条件で、王道的なビール商品のコンセプトを{batch_size}個生成してください。

            【ターゲットペルソナ】
            年齢: {persona.age}
            性別: {persona.gender}
            職業: {persona.occupation}
            ライフスタイル: {persona.lifestyle}
            飲酒習慣: {persona.drinking_habits}
            ビール好み: {persona.beer_preferences}

            【ニードスコープ】
            タイプ: {needscope_info['name']}
            特徴: {needscope_info['description']}
            キーワード: {', '.join(needscope_info['keywords'])}
            """

            # 仮の生成テキスト（本来はAIなどを使用）
            generated_text = f"#1 コンセプト名: ビールA\n説明: 本質を追求したビール\n主な特徴: - 特徴1\n- 特徴2\n想定価格: 300円\n提供価値: 高い"
            idea_blocks = self._split_ideas(generated_text)

            for block in idea_blocks:
                idea = self._parse_idea_block(
                    block, str(uuid.uuid4()), persona.persona_id, needscope_type
                )
                ideas.append(idea)

            logger.info(f"{len(ideas)}個のアイディアを生成完了")
            return ideas

        except Exception as e:
            logger.error(f"アイディア生成エラー: {str(e)}")
            st.error(f"アイディア生成中にエラーが発生しました: {str(e)}")
            return []

    def _split_ideas(self, text: str) -> List[str]:
        """生成されたテキストを個別のアイディアブロックに分割"""
        blocks = []
        current_block = []

        for line in text.split("\n"):
            if line.strip().startswith("#"):
                if current_block:
                    blocks.append("\n".join(current_block))
                current_block = []
            current_block.append(line.strip())

        if current_block:
            blocks.append("\n".join(current_block))

        return blocks

    def _parse_idea_block(
        self,
        block: str,
        session_id: str,
        persona_id: str,
        needscope_type: NeedscopeType,
    ) -> Idea:
        """アイディアブロックをIdea構造体にパース"""
        lines = block.split("\n")
        idea_data = {
            "concept_name": "",
            "description": "",
            "key_features": [],
            "target_price": "",
            "value_proposition": "",
        }

        current_section = None
        for line in lines:
            line = line.strip()
            if line.startswith("コンセプト名:"):
                current_section = "concept_name"
                idea_data[current_section] = line.split(":", 1)[1].strip()
            elif line.startswith("説明:"):
                current_section = "description"
                idea_data[current_section] = line.split(":", 1)[1].strip()
            elif line.startswith("主な特徴:"):
                current_section = "key_features"
            elif line.startswith("想定価格:"):
                current_section = "target_price"
                idea_data[current_section] = line.split(":", 1)[1].strip()
            elif line.startswith("提供価値:"):
                current_section = "value_proposition"
                idea_data[current_section] = line.split(":", 1)[1].strip()
            elif line.startswith("-") and current_section == "key_features":
                idea_data[current_section].append(line[1:].strip())

        evaluation_score = self._calculate_evaluation_score(idea_data, needscope_type)

        return Idea(
            idea_id=str(uuid.uuid4()),
            session_id=session_id,
            persona_id=persona_id,
            needscope_type=needscope_type,
            concept_name=idea_data["concept_name"],
            description=idea_data["description"],
            key_features=idea_data["key_features"],
            target_price=idea_data["target_price"],
            value_proposition=idea_data["value_proposition"],
            evaluation_score=evaluation_score,
            created_at=datetime.datetime.now(),
        )

    def _calculate_evaluation_score(
        self, idea_data: Dict[str, Any], needscope_type: NeedscopeType
    ) -> float:
        """アイディアの評価スコアを計算"""
        score = 0.0
        # 仮のニードスコープデータ
        NEEDSCOPE_DATA = {
            NeedscopeType.STABILITY: {"keywords": ["安定", "持続"]},
            NeedscopeType.HARMONY: {"keywords": ["協調", "共感"]},
            NeedscopeType.BELONGING: {"keywords": ["グループ", "連帯"]},
            NeedscopeType.INDEPENDENCE: {"keywords": ["自由", "冒険"]},
            NeedscopeType.POWER: {"keywords": ["リーダー", "支配"]},
            NeedscopeType.FUN: {"keywords": ["楽しみ", "遊び"]},
        }
        needscope_info = NEEDSCOPE_DATA[needscope_type]

        keyword_matches = sum(
            1
            for keyword in needscope_info["keywords"]
            if keyword in idea_data["description"].lower()
            or keyword in idea_data["value_proposition"].lower()
            or any(keyword in feature.lower() for feature in idea_data["key_features"])
        )

        score += (keyword_matches / len(needscope_info["keywords"])) * 40
        score += len(idea_data["key_features"]) * 10
        score += 30

        return min(100, score)


# カスタムスタイルの定義
CUSTOM_STYLES = """
<style>
    /* メインコンテナ */
    .main-container {
        padding: 2rem;
        background-color: #f8f9fa;
        border-radius: 10px;
        margin-bottom: 2rem;
    }

    /* ヘッダー */
    .beer-storm-header {
        background: linear-gradient(135deg, #F39C12 0%, #F1C40F 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }

    /* カード */
    .idea-card {
        background-color: white;
        padding: 1.5rem;
        border-radius: 8px;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        transition: transform 0.2s ease;
    }

    .idea-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }

    /* バッジ */
    .needscope-badge {
        padding: 0.25rem 0.75rem;
        border-radius: 15px;
        font-size: 0.875rem;
        font-weight: 500;
    }

    /* アニメーション */
    @keyframes storm-effect {
        0% { transform: translateY(0); }
        50% { transform: translateY(-5px); }
        100% { transform: translateY(0); }
    }

    .storm-animate {
        animation: storm-effect 2s infinite;
    }

    /* プログレスバー */
    .stProgress > div > div {
        background-color: #F39C12;
    }

    /* メトリクス */
    .metric-card {
        background-color: white;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
    }

    /* フォーム */
    .stButton>button {
        background-color: #F39C12;
        color: white;
    }

    .stButton>button:hover {
        background-color: #E67E22;
    }
</style>
"""


class UIManager:
    def __init__(self, system: BeerStormSystem):
        self.system = system
        self.setup_page()

    def setup_page(self):
        """ページの基本設定"""
        st.set_page_config(
            page_title="BEER STORM",
            page_icon="🍺",
            layout="wide",
            initial_sidebar_state="expanded",
        )
        st.markdown(CUSTOM_STYLES, unsafe_allow_html=True)

    def render_header(self):
        """ヘッダーの表示"""
        st.markdown(
            """
            <div class="beer-storm-header">
                <h1>🍺 BEER STORM</h1>
                <p>Orthodox Beer Concept Generator 1000</p>
            </div>
        """,
            unsafe_allow_html=True,
        )

    def render_sidebar(self):
        """サイドバーの表示"""
        with st.sidebar:
            st.title("BEER STORM")

            if st.session_state.current_session_id:
                st.success(f"セッションID: {st.session_state.current_session_id}")

            # メニュー選択
            menu = st.radio(
                "機能を選択",
                ["ホーム", "アイディア生成", "詳細分析", "エクスポート", "設定"],
            )

            # 設定パネル
            with st.expander("詳細設定"):
                st.session_state.settings.update(
                    {
                        "batch_size": st.slider(
                            "バッチサイズ",
                            min_value=1,
                            max_value=10,
                            value=st.session_state.settings["batch_size"],
                        ),
                        "temperature": st.slider(
                            "創造性レベル",
                            min_value=0.0,
                            max_value=1.0,
                            value=st.session_state.settings["temperature"],
                        ),
                        "auto_save": st.checkbox(
                            "自動保存", value=st.session_state.settings["auto_save"]
                        ),
                    }
                )

            return menu

    def render_home(self):
        """ホーム画面の表示"""
        self.render_header()

        if not st.session_state.current_session_id:
            st.markdown(
                """
                <div class="main-container">
                    <h2>🌪️ 新しいストームを始めましょう！</h2>
                    <p>ビールのコンセプトアイディアを1000個生成します。</p>
                </div>
            """,
                unsafe_allow_html=True,
            )

            with st.form("new_session_form"):
                session_description = st.text_input(
                    "セッションの説明", placeholder="例：夏季向け新商品のアイディア出し"
                )
                session_tags = st.text_input(
                    "タグ（カンマ区切り）", placeholder="例：夏季,生ビール,期間限定"
                )

                if st.form_submit_button("ストームを開始", use_container_width=True):
                    session_id = self.system.create_session(
                        description=session_description, tags=session_tags
                    )
                    st.session_state.current_session_id = session_id
                    st.experimental_rerun()

        else:
            # セッション情報の表示
            session_info = self.system.get_session_info(
                st.session_state.current_session_id
            )

            col1, col2, col3 = st.columns(3)
            with col1:
                self._render_metric_card(
                    "生成済みアイディア", len(st.session_state.generated_ideas), "個"
                )
            with col2:
                self._render_metric_card(
                    "平均評価スコア", f"{self.system.get_average_score():.1f}", "点"
                )
            with col3:
                self._render_metric_card(
                    "残りアイディア数",
                    1000 - len(st.session_state.generated_ideas),
                    "個",
                )

            # プログレスバー
            progress = len(st.session_state.generated_ideas) / 1000
            st.progress(progress)

    def _render_metric_card(self, title: str, value: Any, unit: str):
        """メトリクスカードの表示"""
        st.markdown(
            f"""
            <div class="metric-card">
                <h3>{title}</h3>
                <h2>{value}{unit}</h2>
            </div>
        """,
            unsafe_allow_html=True,
        )

    def render_idea_generation(self):
        """アイディア生成画面の表示"""
        if not st.session_state.current_session_id:
            st.warning("先にホーム画面でセッションを開始してください")
            return

        st.markdown(
            """
            <div class="main-container">
                <h2>🌪️ アイディアストーム</h2>
            </div>
        """,
            unsafe_allow_html=True,
        )

        # ペルソナ設定
        with st.expander("ペルソナ設定", expanded=not st.session_state.current_persona):
            self._render_persona_section()

        # アイディア生成
        if st.session_state.current_persona:
            with st.form("idea_generation_form"):
                col1, col2 = st.columns(2)

                with col1:
                    needscope_type = st.selectbox(
                        "ニードスコープタイプ",
                        options=[nt for nt in NeedscopeType],
                        format_func=lambda x: NEEDSCOPE_DATA[x]["name"],
                    )

                with col2:
                    batch_size = st.slider(
                        "生成数",
                        min_value=1,
                        max_value=st.session_state.settings["batch_size"],
                        value=5,
                    )

                if st.form_submit_button("アイディアを生成", use_container_width=True):
                    self._generate_ideas(needscope_type, batch_size)

    def _render_persona_section(self):
        """ペルソナセクションの表示"""
        if st.session_state.current_persona:
            self._display_persona(st.session_state.current_persona)
            if st.button("新しいペルソナを生成"):
                st.session_state.current_persona = None
                st.experimental_rerun()
        else:
            if st.button("ペルソナを生成"):
                with st.spinner("ペルソナを生成中..."):
                    try:
                        persona = self.system.generate_persona()
                        st.session_state.current_persona = persona
                        st.experimental_rerun()
                    except Exception as e:
                        st.error(f"ペルソナ生成エラー: {str(e)}")

    def _display_persona(self, persona: Persona):
        """ペルソナ情報の表示"""
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### 基本情報")
            st.write(f"年齢: {persona.age}")
            st.write(f"性別: {persona.gender}")
            st.write(f"職業: {persona.occupation}")
            st.write(f"年収: {persona.income}")

        with col2:
            st.markdown("#### ビール関連情報")
            st.write(f"飲酒習慣: {persona.drinking_habits}")
            st.write(f"好み: {persona.beer_preferences}")
            st.write(f"価格感応度: {persona.price_sensitivity}")

    def _generate_ideas(self, needscope_type: NeedscopeType, batch_size: int):
        """アイディアの生成と表示"""
        with st.spinner(f"{batch_size}個のアイディアを生成中..."):
            try:
                new_ideas = self.system.generate_ideas(
                    st.session_state.current_persona, needscope_type, batch_size
                )

                if new_ideas:
                    st.session_state.generated_ideas.extend(new_ideas)

                    # 生成されたアイディアの表示
                    st.markdown("### 生成されたアイディア")
                    for idea in new_ideas:
                        self._display_idea_card(idea)

                    # 進捗の更新
                    if len(st.session_state.generated_ideas) >= 1000:
                        st.balloons()
                        st.success("🎉 1000個のアイディア生成が完了しました！")

            except Exception as e:
                st.error(f"アイディア生成エラー: {str(e)}")

    def _display_idea_card(self, idea: Idea):
        """アイディアカードの表示"""
        with st.container():
            st.markdown(
                f"""
                <div class="idea-card">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <h3>{idea.concept_name}</h3>
                        <span class="needscope-badge" style="background-color: {NEEDSCOPE_DATA[idea.needscope_type]['color']}">
                            {NEEDSCOPE_DATA[idea.needscope_type]['name']}
                        </span>
                    </div>
                    <p>{idea.description}</p>
                    <h4>主な特徴:</h4>
                    <ul>
                        {''.join([f'<li>{feature}</li>' for feature in idea.key_features])}
                    </ul>
                    <div style="display: flex; justify-content: space-between;">
                        <span>想定価格: {idea.target_price}</span>
                        <span>評価スコア: {idea.evaluation_score:.1f}</span>
                    </div>
                </div>
            """,
                unsafe_allow_html=True,
            )


class UIManager:
    # Part 3の続き...

    def render_analysis(self):
        """分析画面の表示"""
        st.markdown(
            """
            <div class="main-container">
                <h2>📊 詳細分析</h2>
            </div>
        """,
            unsafe_allow_html=True,
        )

        # セッション選択
        sessions = self.system.list_sessions()
        selected_session = st.selectbox(
            "分析するセッションを選択",
            options=[None] + sessions,
            format_func=lambda x: (
                "現在のセッション" if x is None else f"セッション: {x}"
            ),
        )

        if not selected_session and not st.session_state.current_session_id:
            st.warning("分析するセッションがありません")
            return

        session_id = selected_session or st.session_state.current_session_id

        # 分析タイプの選択
        analysis_types = st.multiselect(
            "分析項目を選択",
            options=[
                "ニードスコープ分布",
                "評価スコア分析",
                "キーワード分析",
                "価格帯分析",
                "トレンド分析",
            ],
            default=["ニードスコープ分布", "評価スコア分析"],
        )

        if st.button("分析を実行", use_container_width=True):
            with st.spinner("分析を実行中..."):
                try:
                    results = self.system.analyze_session_results(session_id)
                    self._display_analysis_results(results, analysis_types)
                except Exception as e:
                    st.error(f"分析エラー: {str(e)}")

    def _display_analysis_results(
        self, results: Dict[str, Any], selected_types: List[str]
    ):
        """分析結果の表示"""
        # サマリー統計
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            self._render_metric_card(
                "総アイディア数", results["summary"]["total_ideas"], "個"
            )
        with col2:
            self._render_metric_card(
                "平均評価スコア", f"{results['summary']['average_score']:.1f}", "点"
            )
        with col3:
            self._render_metric_card(
                "高評価アイディア", results["summary"]["high_score_ideas"], "個"
            )
        with col4:
            self._render_metric_card(
                "使用ペルソナ数", results["summary"]["unique_personas"], "人"
            )

        # 選択された分析の表示
        for analysis_type in selected_types:
            with st.expander(analysis_type, expanded=True):
                if analysis_type == "ニードスコープ分布":
                    st.plotly_chart(
                        results["needscope_distribution"]["visualization"],
                        use_container_width=True,
                    )

                elif analysis_type == "評価スコア分析":
                    st.plotly_chart(
                        results["evaluation_scores"]["visualization"],
                        use_container_width=True,
                    )

                elif analysis_type == "キーワード分析":
                    col1, col2 = st.columns(2)
                    with col1:
                        st.plotly_chart(
                            results["keyword_analysis"]["wordcloud"],
                            use_container_width=True,
                        )
                    with col2:
                        st.dataframe(results["keyword_analysis"]["frequency_table"])

                elif analysis_type == "価格帯分析":
                    st.plotly_chart(
                        results["price_analysis"]["visualization"],
                        use_container_width=True,
                    )

                elif analysis_type == "トレンド分析":
                    st.plotly_chart(
                        results["trends"]["visualization"], use_container_width=True
                    )

    def render_export(self):
        """エクスポート画面の表示"""
        st.markdown(
            """
            <div class="main-container">
                <h2>📤 データエクスポート</h2>
            </div>
        """,
            unsafe_allow_html=True,
        )

        # セッション選択
        sessions = self.system.list_sessions()
        selected_session = st.selectbox(
            "エクスポートするセッションを選択",
            options=[None] + sessions,
            format_func=lambda x: (
                "現在のセッション" if x is None else f"セッション: {x}"
            ),
        )

        if not selected_session and not st.session_state.current_session_id:
            st.warning("エクスポートするセッションがありません")
            return

        session_id = selected_session or st.session_state.current_session_id

        # エクスポート設定
        col1, col2 = st.columns(2)
        with col1:
            export_format = st.selectbox(
                "エクスポート形式", options=["Excel", "CSV", "JSON"]
            )

        with col2:
            export_contents = st.multiselect(
                "エクスポート内容",
                options=["ペルソナデータ", "アイディアデータ", "分析結果"],
                default=["アイディアデータ"],
            )

        if st.button("エクスポートを実行", use_container_width=True):
            with st.spinner("データをエクスポート中..."):
                try:
                    export_data = self.system.export_session_data(
                        session_id, export_format, export_contents
                    )

                    # エクスポートファイルのダウンロード提供
                    if export_format == "Excel":
                        st.download_button(
                            label="Excelファイルをダウンロード",
                            data=export_data,
                            file_name=f"beer_storm_export_{session_id}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        )

                    elif export_format == "CSV":
                        for content_type, csv_data in export_data.items():
                            st.download_button(
                                label=f"{content_type}のCSVをダウンロード",
                                data=csv_data,
                                file_name=f"beer_storm_{content_type}_{session_id}.csv",
                                mime="text/csv",
                            )

                    else:  # JSON
                        st.download_button(
                            label="JSONファイルをダウンロード",
                            data=export_data,
                            file_name=f"beer_storm_export_{session_id}.json",
                            mime="application/json",
                        )

                except Exception as e:
                    st.error(f"エクスポートエラー: {str(e)}")


def main():
    """メインアプリケーション"""
    try:
        # システムの初期化
        if "system" not in st.session_state:
            st.session_state.system = BeerStormSystem()

        # UIマネージャーの初期化
        ui_manager = UIManager(st.session_state.system)

        # メニュー選択
        selected_menu = ui_manager.render_sidebar()

        # 選択された画面の表示
        if selected_menu == "ホーム":
            ui_manager.render_home()
        elif selected_menu == "アイディア生成":
            ui_manager.render_idea_generation()
        elif selected_menu == "詳細分析":
            ui_manager.render_analysis()
        elif selected_menu == "エクスポート":
            ui_manager.render_export()
        else:  # 設定
            ui_manager.render_settings()

        # エラーログの表示（開発モード時のみ）
        if st.session_state.settings.get("dev_mode"):
            with st.expander("エラーログ"):
                for error in st.session_state.error_log:
                    st.error(error)

    except Exception as e:
        logger.error(f"アプリケーションエラー: {str(e)}")
        st.error("予期せぬエラーが発生しました。更新してやり直してください。")


if __name__ == "__main__":
    main()
