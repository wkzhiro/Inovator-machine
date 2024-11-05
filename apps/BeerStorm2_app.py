from abc import ABC, abstractmethod
import streamlit as st
from dataclasses import dataclass
import pandas as pd
import datetime
import uuid
import logging
import time
import os
import json
from dotenv import load_dotenv
from typing import Dict, List, Optional, Union, Any, TYPE_CHECKING
from enum import Enum
import openai
import random
import re
import gspread
from google.oauth2.service_account import Credentials
from typing import Dict, List, Optional, Union, Any, TYPE_CHECKING, TypedDict

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("app.log")],
)
logger = logging.getLogger(__name__)

idea_size=int(os.getenv("IDEA_SIZE",100))

class APIError(Exception):
    """API操作に関する基本的なエラー"""

    pass


class RateLimitError(APIError):
    """レート制限に関するエラー"""

    pass


# レート制限管理クラス
class RateLimiter:
    """APIレート制限を管理するクラス"""

    def __init__(self, calls_per_minute: int = 60, max_retries: int = 3):
        self.calls_per_minute = calls_per_minute
        self.max_retries = max_retries
        self.calls: List[float] = []
        self._last_cleanup: float = time.time()

    def wait_if_needed(self) -> None:
        now = time.time()
        if now - self._last_cleanup >= 60:
            self._cleanup_old_calls(now)
            self._last_cleanup = now

        retries = 0
        while len(self.calls) >= self.calls_per_minute and retries < self.max_retries:
            sleep_time = 60 - (now - self.calls[0])
            if sleep_time > 0:
                logger.debug(f"Rate limit reached. Waiting {sleep_time:.2f} seconds.")
                time.sleep(sleep_time)
                retries += 1
            now = time.time()
            self._cleanup_old_calls(now)

        if len(self.calls) >= self.calls_per_minute:
            raise RateLimitError("Rate limit exceeded after maximum retries")

        self.calls.append(now)

    def _cleanup_old_calls(self, current_time: float) -> None:
        self.calls = [
            call_time for call_time in self.calls if current_time - call_time <= 60
        ]


class APIClient(ABC):
    """API操作の基底クラス"""

    def __init__(self, config: "APIConfig"):
        self.config = config
        self.rate_limiter = RateLimiter()

    def _handle_api_error(self, e: Exception, operation: str) -> None:
        error_msg = f"API error during {operation}: {str(e)}"
        logger.error(error_msg)
        raise APIError(error_msg) from e

    @abstractmethod
    def _setup_connection(self) -> None:
        """API接続のセットアップ（サブクラスで実装）"""
        pass


class NeedscopeAttributes(TypedDict):
    """ニードスコープの属性を定義する型"""

    name: str
    color: str
    keywords: List[str]
    description: str
    weight: float
    # category: str
    emotional_value: str
    # target_demographic: str
    # consumption_occasion: str


class NeedscopeType(Enum):
    """ニードスコープの種類を定義するクラス"""

    STIMULATION = "刺激"
    ADMIRATION = "権威"
    MASTERY = "洗練"
    UNWIND = "癒し"
    CONNECTION = "陽気"
    RELEASE = "楽しさ"

    @property
    def display_name(self) -> str:
        """表示用の名前を返す"""
        return NEEDSCOPE_DATA[self]["name"]

    @property
    def color(self) -> str:
        """カラーコードを返す"""
        return NEEDSCOPE_DATA[self]["color"]

    @property
    def keywords(self) -> List[str]:
        """キーワードリストを返す"""
        return NEEDSCOPE_DATA[self]["keywords"]

    @property
    def description(self) -> str:
        """説明文を返す"""
        return NEEDSCOPE_DATA[self]["description"]

    @property
    def category(self) -> str:
        """カテゴリーを返す"""
        return NEEDSCOPE_DATA[self]["category"]

    @property
    def emotional_value(self) -> str:
        """感情的価値を返す"""
        return NEEDSCOPE_DATA[self]["emotional_value"]

    def get_evaluation_weight(self) -> float:
        """評価時の重み付けを返す"""
        return NEEDSCOPE_DATA[self]["weight"]

    def matches_keywords(self, text: str) -> bool:
        """テキストがニードスコープのキーワードと一致するかチェック"""
        text_lower = text.lower()
        return any(keyword.lower() in text_lower for keyword in self.keywords)

    @classmethod
    def from_japanese(cls, japanese_name: str) -> "NeedscopeType":
        """日本語名からニードスコープタイプを取得"""
        for needscope in cls:
            if needscope.value == japanese_name:
                return needscope
        raise ValueError(f"Invalid needscope name: {japanese_name}")

    @classmethod
    def get_all_keywords(cls) -> List[str]:
        """全てのニードスコープのキーワードを取得"""
        all_keywords = []
        for needscope in cls:
            all_keywords.extend(needscope.keywords)
        return list(set(all_keywords))


# NEEDSCOPEデータの定義
NEEDSCOPE_DATA: Dict[NeedscopeType, NeedscopeAttributes] = {
    NeedscopeType.STIMULATION: {
        "name": "刺激 (stimulation)",
        "color": "#ff0000",
        "keywords": [
            "刺激",
            "エッジ",
            "強い",
            "ユニーク",
            "アクティブ",
            "ダイナミック",
            "革新的",
            "先進的",
            "挑戦的",
        ],
        "description": "強く鋭い口当たり、炭酸が強く、アルコール度数が高いビールを求める",
        "weight": 1.0,
        # "category": "革新志向",
        "emotional_value": "活力、新しい体験への期待",
        # "target_demographic": "トレンドに敏感な若年層",
        # "consumption_occasion": "特別な機会やイベント",
    },
    NeedscopeType.ADMIRATION: {
        "name": "権威 (Admiration)",
        "color": "#960046",
        "keywords": [
            "威厳",
            "あこがれ",
            "最高級",
            "豪華",
            "特別",
            "芸術的",
            "伝統",
            "格式",
            "洗練",
        ],
        "description": "最高級の食材で作られた、濃厚で重くて風味豊かなビールを求める",
        "weight": 1.2,
        # "category": "プレステージ志向",
        "emotional_value": "優越感と満足感",
        # "target_demographic": "富裕層・経営者層",
        # "consumption_occasion": "高級レストラン・特別な祝い事",
    },
    NeedscopeType.MASTERY: {
        "name": "洗練 (Mastery)",
        "color": "#006400",
        "keywords": [
            "洗練",
            "品質",
            "伝統",
            "熟成",
            "技巧",
            "深み",
            "匠",
            "職人",
            "本物",
        ],
        "description": "伝統的な製法で作られた、深い味わいと香りのビールを求める",
        "weight": 1.1,
        # "category": "本物志向",
        "emotional_value": "味わいの深さへの探求",
        # "target_demographic": "ビール通・愛好家",
        # "consumption_occasion": "じっくりと味わう時間",
    },
    NeedscopeType.UNWIND: {
        "name": "癒し (Unwind)",
        "color": "#0000ff",
        "keywords": [
            "癒し",
            "リラックス",
            "穏やか",
            "自然",
            "スムーズ",
            "バランス",
            "心地よさ",
            "やすらぎ",
        ],
        "description": "滑らかで飲みやすく、リラックスできるビールを求める",
        "weight": 0.9,
        # "category": "リラックス志向",
        "emotional_value": "心身のリフレッシュ",
        # "target_demographic": "ストレス社会で生きる現代人",
        # "consumption_occasion": "リラックスタイム・休日",
    },
    NeedscopeType.CONNECTION: {
        "name": "陽気 (Connection)",
        "color": "#ff69b4",
        "keywords": [
            "陽気",
            "楽しい",
            "社交的",
            "カジュアル",
            "明るい",
            "フレンドリー",
            "にぎやか",
            "交流",
        ],
        "description": "仲間と楽しめる、軽快で飲みやすいビールを求める",
        "weight": 0.8,
        # "category": "コミュニケーション志向",
        "emotional_value": "共に楽しむ喜び",
        # "target_demographic": "社交的な若年～中年層",
        # "consumption_occasion": "パーティー・飲み会",
    },
    NeedscopeType.RELEASE: {
        "name": "楽しさ (Release)",
        "color": "#ffa500",
        "keywords": [
            "楽しさ",
            "解放",
            "エネルギー",
            "冒険",
            "大胆",
            "刺激的",
            "遊び心",
            "開放的",
        ],
        "description": "新しい体験や刺激を求める、ユニークなビールを求める",
        "weight": 0.95,
        # "category": "エンターテインメント志向",
        "emotional_value": "解放感と高揚感",
        # "target_demographic": "アクティブな若年層",
        # "consumption_occasion": "レジャー・アウトドア",
    },
}


@dataclass
class NeedscopeAnalysis:
    """ニードスコープ分析結果を保持するクラス"""

    needscope_type: NeedscopeType
    match_score: float
    keyword_matches: List[str]
    emotional_alignment: float
    target_fit: float

    @property
    def total_score(self) -> float:
        """総合スコアを計算"""
        return (
            self.match_score * 0.4
            + len(self.keyword_matches) * 0.3
            + self.emotional_alignment * 0.2
            + self.target_fit * 0.1
        ) * self.needscope_type.get_evaluation_weight()


def analyze_text_for_needscope(text: str) -> List[NeedscopeAnalysis]:
    """テキストのニードスコープ分析を行う"""
    results = []

    for needscope in NeedscopeType:
        # キーワードマッチング
        matched_keywords = [
            keyword for keyword in needscope.keywords if keyword.lower() in text.lower()
        ]

        # 基本的なマッチスコア計算
        match_score = len(matched_keywords) / len(needscope.keywords)

        # 感情的一致度の簡易計算（実際にはより複雑なアルゴリズムを使用）
        emotional_alignment = 0.5  # デフォルト値

        # ターゲット適合度の簡易計算
        target_fit = 0.5  # デフォルト値

        analysis = NeedscopeAnalysis(
            needscope_type=needscope,
            match_score=match_score,
            keyword_matches=matched_keywords,
            emotional_alignment=emotional_alignment,
            target_fit=target_fit,
        )
        results.append(analysis)

    # スコアの高い順にソート
    results.sort(key=lambda x: x.total_score, reverse=True)
    return results


@dataclass(frozen=True)
class APIConfig:
    """API設定を保持する不変クラス"""

    azure_endpoint: str
    azure_api_key: str
    azure_model: str
    azure_api_version: str
    google_sheets_id: str
    google_credentials_path: str

    @classmethod
    def from_env(cls) -> "APIConfig":
        """環境変数から設定を読み込む"""
        required_vars = {
            "AZURE_CHAT_ENDPOINT": "Azure APIのエンドポイント",
            "AZURE_API_KEY": "Azure APIキー",
            "AZURE_DEPLOYMENT_NAME": "Azureデプロイメント名",
            "AZURE_API_VERSION": "Azure APIバージョン",
            "GOOGLE_SHEETS_ID": "GoogleスプレッドシートID",
            "GOOGLE_CREDENTIALS_PATH": "Google認証情報のパス",
        }

        # 必要な環境変数の検証
        missing_vars = [
            var for var, desc in required_vars.items() if not os.getenv(var)
        ]
        if missing_vars:
            raise ValueError(
                f"Missing environment variables: {', '.join(missing_vars)}"
            )

        endpoint = os.getenv("AZURE_CHAT_ENDPOINT", "")
        base_endpoint = endpoint.split("/openai/deployments")[0]
        google_creds_path = os.getenv("GOOGLE_CREDENTIALS_PATH")

        if not os.path.exists(google_creds_path):
            raise FileNotFoundError(
                f"Google credentials file not found: {google_creds_path}"
            )

        return cls(
            azure_endpoint=base_endpoint,
            azure_api_key=os.getenv("AZURE_API_KEY"),
            azure_model=os.getenv("AZURE_DEPLOYMENT_NAME"),
            azure_api_version=os.getenv("AZURE_API_VERSION"),
            google_sheets_id=os.getenv("GOOGLE_SHEETS_ID"),
            google_credentials_path=google_creds_path,
        )


@dataclass
class Session:
    """セッション情報を管理するデータクラス"""

    session_id: str
    user_id: str
    start_time: datetime
    end_time: Optional[datetime.datetime]
    status: str
    total_ideas: int
    settings: Dict[str, Any]


@dataclass
class User:
    """ユーザー情報を管理するデータクラス"""

    user_id: str
    name: str
    email: str
    department: str
    created_at: datetime


@dataclass
class SessionState:
    """セッション状態を管理するクラス"""

    settings: Dict[str, Any]
    personas: List["Persona"]
    generated_ideas: List["Idea"]
    current_batch_index: int
    personas_confirmed: bool
    current_session_id: str

    @classmethod
    def initialize(cls) -> "SessionState":
        """新しいセッション状態を初期化"""
        return cls(
            settings={
                "dev_mode": False,
                "temperature": 0.7,
                "batch_size": 10,
                "auto_save": True,
            },
            personas=[],
            generated_ideas=[],
            current_batch_index=0,
            personas_confirmed=False,
            current_session_id=str(uuid.uuid4()),
        )


@dataclass
class Persona:
    """ペルソナを表すデータクラス"""

    persona_id: str
    age: int
    gender: str
    occupation: str
    lifestyle: str
    thinking_style: str
    work_style: str
    drinking_habits: str
    beer_preferences: str
    recent_concerns:str
    family_composition:str
    information_gathering_methods:str
    health_consciousness:str
    community:str
    criteria_shopping_decisions:str
    leisure_time_usage:str

    def to_dict(self) -> Dict[str, Any]:
        """ペルソナをディクショナリに変換"""
        return {
            "persona_id": self.persona_id,
            "age": self.age,
            "gender": self.gender,
            "occupation": self.occupation,
            # "expertise": self.expertise,
            # "experience": self.experience,
            "thinking_style": self.thinking_style,
            "work_style": self.work_style,
            "lifestyle": self.lifestyle,
            "drinking_habits": self.drinking_habits,
            "beer_preferences": self.beer_preferences,
            "recent_concerns":self.recent_concerns,
            "family_composition":self.family_composition,
            "information_gathering_methods":self.information_gathering_methods,
            "health_consciousness":self.health_consciousness,
            "community":self.community,
            "criteria_shopping_decisions":self.criteria_shopping_decisions,
            "leisure_time_usage":self.leisure_time_usage,
        }


@dataclass
class Idea:
    """ビールコンセプトのアイデア情報"""

    idea_id: str
    session_id: str
    persona_id: str
    needscope_type: NeedscopeType
    concept_name: str
    features: str
    price: str
    description: str
    evaluation_score: float
    tagline:str
    accepted_consumer_belief:str
    reason_to_believe:str

    created_at: datetime.datetime

    def to_dict(self) -> Dict[str, Any]:
        """アイデアをディクショナリに変換"""
        return {
            "idea_id": self.idea_id,
            "session_id": self.session_id,
            "persona_id": self.persona_id,
            "needscope_type": self.needscope_type.value,
            "concept_name": self.concept_name,
            "description": self.description,
            "features":self.features,
            "price":self.price,
            "tagline":self.tagline,
            "accepted_consumer_belief":self.accepted_consumer_belief,
            "reason_to_believe":self.reason_to_believe,
            "evaluation_score": self.evaluation_score,
            "created_at": self.created_at.isoformat(),
        }


def init_session_state():
    """セッション状態の初期化"""
    if "settings" not in st.session_state:
        st.session_state.settings = {
            "dev_mode": False,
            "temperature": 0.7,
            "batch_size": 10,
            "auto_save": True,
        }
    if "personas" not in st.session_state:
        st.session_state.personas = []
    if "generated_ideas" not in st.session_state:
        st.session_state.generated_ideas = []
    if "current_batch_index" not in st.session_state:
        st.session_state.current_batch_index = 0
    if "personas_confirmed" not in st.session_state:
        st.session_state.personas_confirmed = False
    if "current_session_id" not in st.session_state:
        st.session_state.current_session_id = str(uuid.uuid4())


class OpenAIClient(APIClient):
    """OpenAI API操作を管理するクラス"""

    def __init__(self, config: "APIConfig"):
        super().__init__(config)
        self.client = None  # インスタンス変数として初期化
        self._setup_connection()

    def _setup_connection(self) -> None:
        """API接続の設定を行うメソッド"""
        self._setup_client()

    def _setup_client(self) -> None:
        """OpenAI APIの設定"""
        try:
            self.client = openai.AzureOpenAI(
                api_key=self.config.azure_api_key,
                api_version=self.config.azure_api_version,
                azure_endpoint=self.config.azure_endpoint
            )
            # openai.api_type = "azure"
            # openai.api_base = self.config.azure_endpoint
            # openai.api_version = self.config.azure_api_version
            # openai.api_key = self.config.azure_api_key
            logger.info("OpenAI API configured successfully")
            # print(f"OpenAI API Base URL: {openai.api_base}")

        except Exception as e:
            self._handle_api_error(e, "OpenAI setup")

    def generate_completion(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """
        ChatGPT APIを使用して応答を生成するメソッド

        引数:
            messages (List[Dict[str, str]]): 会話メッセージのリスト
            **kwargs: その他のオプションパラメータ

        戻り値:
            str: 生成されたテキスト

        例外:
            APIError: API呼び出しに失敗した場合
        """
        try:
            # レート制限をチェックし、必要であれば待機
            self.rate_limiter.wait_if_needed()
            # チャット応答の生成
            response = self.client.chat.completions.create(
                model=self.config.azure_model,
                messages=messages,
                response_format={"type": "json_object"},  # JSONフォーマットを指定
                **kwargs
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            self._handle_api_error(e, "チャット応答生成")

@dataclass
class WorksheetDataCacheEntry:
    data: List[Dict[str, Any]]
    last_fetched: float

class GoogleSheetsClient(APIClient):
    def __init__(self, config: APIConfig):
        super().__init__(config)
        self.spreadsheet = self._setup_connection()
        self._worksheet_cache = {}
        self._ensure_worksheets_exist()
        self._worksheet_data_cache: Dict[str, WorksheetDataCacheEntry] = {}
        # キャッシュの有効期限（秒）
        self.data_cache_expiry_seconds = 300  # 5分

    def _setup_connection(self) -> Any:
        try:
            credentials_info = os.getenv("GOOGLE_CREDENTIALS_JSON")
            if not credentials_info:
                raise ValueError("環境変数 'GOOGLE_CREDENTIALS_JSON' が設定されていません")

            credentials = Credentials.from_service_account_info(
                json.loads(credentials_info),
                # self.config.google_credentials_path,
                scopes=["https://www.googleapis.com/auth/spreadsheets"],
            )
            gc = gspread.authorize(credentials)
            spreadsheet = gc.open_by_key(self.config.google_sheets_id)
            logger.info("Successfully connected to Google Spreadsheet")
            return spreadsheet
        except Exception as e:
            raise APIError(f"Failed to setup Google Sheets connection: {str(e)}")

    def _ensure_worksheets_exist(self):
        """Ensure required worksheets exist, create if they don't"""
        required_worksheets = {
            "sessions": [
                "session_id",
                "user_id",
                "start_time",
                "end_time",
                "status",
                "total_ideas",
                "settings",
            ],
            "ideas": [
                "idea_id",
                "session_id",
                "persona_id",
                "needscope_type",
                "concept_name",
                "description",
                "features",
                "price",
                "tagline",
                "accepted_consumer_belief",
                "reason_to_believe",
                "evaluation_score",
                "created_at",
            ],
        }

        try:
            existing_worksheets = [ws.title for ws in self.spreadsheet.worksheets()]

            for worksheet_name, headers in required_worksheets.items():
                if worksheet_name not in existing_worksheets:
                    worksheet = self.spreadsheet.add_worksheet(
                        title=worksheet_name, rows=idea_size, cols=str(len(headers))
                    )
                    worksheet.append_row(headers)
                    logger.info(f"Created worksheet: {worksheet_name}")
                    self._worksheet_cache[worksheet_name] = worksheet
                else:
                    worksheet = self.spreadsheet.worksheet(worksheet_name)
                    current_headers = worksheet.row_values(1)
                    if current_headers != headers:
                        worksheet.clear()
                        worksheet.append_row(headers)
                        logger.info(f"Reset headers for worksheet: {worksheet_name}")
                    self._worksheet_cache[worksheet_name] = worksheet

        except Exception as e:
            logger.error(f"Failed to ensure worksheets exist: {str(e)}")
            raise APIError(f"Worksheet setup failed: {str(e)}")

    def append_row(self, worksheet_name: str, row_data: List[Any]) -> None:
        """Append a row of data to specified worksheet with error handling and retries"""
        max_retries = 5
        current_retry = 0
        base_wait_time = 2

        while current_retry < max_retries:
            try:
                # Rate limiter wait
                self.rate_limiter.wait_if_needed()

                # Ensure worksheet exists and is cached
                if worksheet_name not in self._worksheet_cache:
                    self._ensure_worksheets_exist()

                worksheet = self._worksheet_cache[worksheet_name]

                # Format data
                formatted_data = [
                    str(item) if item is not None else "" for item in row_data
                ]

                # Verify data length matches headers
                headers = worksheet.row_values(1)
                if len(formatted_data) != len(headers):
                    raise ValueError(
                        f"Data length mismatch: expected {len(headers)} fields but got {len(formatted_data)} fields for {worksheet_name}"
                    )

                # Append row
                worksheet.append_row(formatted_data)
                # 行の追加が成功したら、該当ワークシートのデータキャッシュを無効化
                if worksheet_name in self._worksheet_data_cache:
                    del self._worksheet_data_cache[worksheet_name]
                logger.info(
                    f"Successfully appended row to {worksheet_name}: {row_data}"
                )
                return

            except gspread.exceptions.APIError as e:
                current_retry += 1
                if "Quota exceeded" in str(e):
                    wait_time = base_wait_time * (
                        2**current_retry
                    )  # Exponential backoff
                    logger.warning(
                        f"Quota exceeded, retrying in {wait_time} seconds... (attempt {current_retry}/{max_retries})"
                    )
                    time.sleep(wait_time)
                    continue
                else:
                    logger.error(f"Failed to append row due to API error: {e}")
                    raise APIError(f"API error appending row: {str(e)}") from e

            except ValueError as ve:
                logger.error(f"Data validation error: {ve}")
                raise ve

            except Exception as e:
                current_retry += 1
                logger.error(
                    f"An unexpected error occurred on attempt {current_retry}/{max_retries}: {e}"
                )
                if current_retry == max_retries:
                    logger.error(f"Failed to append row after {max_retries} retries.")
                    raise APIError(
                        f"Failed to append row after retries: {str(e)}"
                    ) from e
                time.sleep(base_wait_time)

    def save_idea(self, idea: Idea) -> None:
        """Save an idea to the ideas worksheet"""
        try:
            row_data = [
                str(idea.idea_id),
                str(idea.session_id),
                str(idea.persona_id),
                idea.needscope_type.value,
                str(idea.concept_name),
                str(idea.description),
                str(idea.evaluation_score),
                idea.created_at.isoformat(),
            ]
            self.append_row("ideas", row_data)
        except Exception as e:
            logger.error(f"Failed to save idea: {str(e)}")
            raise APIError(f"Failed to save idea: {str(e)}")

    def _get_all_rows_by_field(
        self, worksheet_name: str, field_value: str, field_index: int
    ) -> List[Dict[str, Any]]:
        try:
            now = time.time()
            cache_entry = self._worksheet_data_cache.get(worksheet_name)

            if cache_entry and now - cache_entry.last_fetched < self.data_cache_expiry_seconds:
                rows = cache_entry.data
            else:
                worksheet, headers = self._get_worksheet(worksheet_name)
                rows = worksheet.get_all_records()
                self._worksheet_data_cache[worksheet_name] = WorksheetDataCacheEntry(
                    data=rows, last_fetched=now
                )

            matching_rows = [
                row
                for row in rows
                if str(row.get(headers[field_index])) == field_value
            ]
            logger.info(f"Filtered rows for {field_value}: {matching_rows}")
            return matching_rows
        except Exception as e:
            logger.error(f"Failed to retrieve rows: {str(e)}")
            return []

    def save_personas(self, personas: List[Persona]) -> None:
        self.sheets_client.save_personas_to_sheet(personas)

        try:
            worksheet_name = "personas"
            headers = [
                "persona_id",
                "age",
                "gender",
                "occupation",
                # "expertise",
                # "experience",
                "thinking_style",
                "work_style",
                "lifestyle",
                "drinking_habits",
                "beer_preferences",
                "recent_concerns",
                "family_composition",
                "information_gathering_methods",
                "health_consciousness",
                "community",
                "criteria_shopping_decisions",
                "leisure_time_usage"
            ]

            # シートが存在しなければ作成
            if worksheet_name not in self._worksheet_cache:
                worksheet = self.spreadsheet.add_worksheet(
                    title=worksheet_name, rows=idea_size, cols=str(len(headers))
                )
                worksheet.append_row(headers)
                self._worksheet_cache[worksheet_name] = worksheet

            worksheet = self._worksheet_cache[worksheet_name]

            # ペルソナごとの情報を行として追加
            for persona in personas:
                row_data = [
                    persona.persona_id,
                    persona.age,
                    persona.gender,
                    persona.occupation,
                    # persona.expertise,
                    # persona.experience,
                    persona.thinking_style,
                    persona.work_style,
                    persona.lifestyle,
                    persona.drinking_habits,
                    persona.beer_preferences,
                    persona.recent_concerns,
                    persona.family_composition,
                    persona.information_gathering_methods,
                    persona.health_consciousness,
                    persona.community,
                    persona.criteria_shopping_decisions,
                    persona.leisure_time_usage
                ]
                worksheet.append_row(row_data)
            logger.info(f"{len(personas)}件のペルソナ情報を保存しました")
        except Exception as e:
            logger.error(f"ペルソナ情報の保存エラー: {str(e)}")
            raise

    def save_execution(self, user_name: str, execution_id: str) -> None:
        self.sheets_client.save_execution_to_sheet(execution_id, user_name)
        try:
            worksheet_name = "executions"
            headers = ["execution_id", "user_name", "timestamp"]

            # シートが存在しなければ作成
            if worksheet_name not in self._worksheet_cache:
                worksheet = self.spreadsheet.add_worksheet(
                    title=worksheet_name, rows=idea_size, cols=str(len(headers))
                )
                worksheet.append_row(headers)
                self._worksheet_cache[worksheet_name] = worksheet

            worksheet = self._worksheet_cache[worksheet_name]

            # 実行情報を行として追加
            timestamp = datetime.datetime.now().isoformat()
            row_data = [execution_id, user_name, timestamp]
            worksheet.append_row(row_data)
            logger.info(f"実行者情報を保存しました: {user_name} ({execution_id})")
        except Exception as e:
            logger.error(f"実行者情報の保存エラー: {str(e)}")
            raise

    def _get_all_rows_by_field(
        self, worksheet_name: str, field_value: str, field_index: int
    ) -> List[Dict[str, Any]]:
        try:
            worksheet = self._worksheet_cache[worksheet_name]
            rows = worksheet.get_all_records()
            matching_rows = [
                row
                for row in rows
                if str(row.get(worksheet.row_values(1)[field_index])) == field_value
            ]
            logger.info(f"Filtered rows for {field_value}: {matching_rows}")
            time.sleep(2)  # API呼び出し間に2秒の待機を追加
            return matching_rows
        except Exception as e:
            logger.error(f"Failed to retrieve rows: {str(e)}")
            return []


class SpreadsheetDB:
    """スプレッドシートをデータベースとして使用するクラス"""

    def __init__(self, sheets_client: GoogleSheetsClient):
        self.sheets_client = sheets_client

    def create_session(self, user_id: str, settings: Dict[str, Any]) -> str:
        """新しいセッションを作成"""
        session_id = str(uuid.uuid4())
        session_data = [
            session_id,
            user_id,
            datetime.datetime.now().isoformat(),  # 開始時刻
            "",  # 終了時刻は初期値として空
            "active",  # セッションの初期状態
            0,  # 初期のアイディア数
            json.dumps(settings),
        ]
        try:
            self.sheets_client.append_row("sessions", session_data)
            return session_id
        except Exception as e:
            logger.error(f"Failed to create session: {str(e)}")
            raise APIError(f"Failed to create session: {str(e)}")

    def get_session_results(self, session_id: str) -> Dict[str, Any]:
        """セッションIDを使用してセッションの結果を取得"""
        try:
            results = {
                "session": None,
                "ideas": [],
            }

            # セッションデータを取得
            session_data = self.sheets_client._get_all_rows_by_field(
                "sessions", session_id, 0  # '0'はsession_idがあるカラムのインデックス
            )
            if session_data:
                results["session"] = Session(
                    session_id=session_id,
                    user_id=session_data[0].get("user_id", ""),
                    start_time=datetime.datetime.fromisoformat(
                        session_data[0].get("start_time", "")
                    ),
                    end_time=(
                        datetime.datetime.fromisoformat(
                            session_data[0].get("end_time", "")
                        )
                        if session_data[0].get("end_time")
                        else None
                    ),
                    status=session_data[0].get("status", ""),
                    total_ideas=int(session_data[0].get("total_ideas", 0)),
                    settings=json.loads(session_data[0].get("settings", "{}")),
                )

            # アイデアデータを取得し、必要な列（description, needscope_type, concept_name）を抽出
            ideas_data = self.sheets_client._get_all_rows_by_field(
                "ideas", session_id, 1  # '1'はsession_idがあるカラムのインデックス
            )
            results["ideas"] = [
                {
                    "description": row.get("description", ""),
                    "needscope_type": row.get("needscope_type", ""),
                    "concept_name": row.get("concept_name", ""),
                }
                for row in ideas_data
            ]

            return results

        except Exception as e:
            logger.error(f"Failed to get session results: {str(e)}")
            return {"session": None, "ideas": []}


class GoogleSheetsError(Exception):
    """Google Sheets操作の基本エラー"""

    pass


class GoogleSheetsAPIError(GoogleSheetsError):
    """Google Sheets API特有のエラー"""

    pass


class BeerStormSystem:
    """アプリケーションのメインシステムクラス"""

    def __init__(self, api_config: APIConfig):
        """
        システムの初期化

        Args:
            api_config: API設定
        """
        self.config = api_config
        self.sheets_client = GoogleSheetsClient(api_config)
        self.spreadsheet_db = SpreadsheetDB(self.sheets_client)  # これを追加
        self.ai_manager = AIManager(api_config)
        self._setup_openai()
        self._setup_spreadsheet()

    def _setup_openai(self):
        """OpenAI APIの設定を初期化"""
        try:
            openai.api_type = "azure"
            openai.api_base = self.config.azure_endpoint
            openai.api_version = self.config.azure_api_version
            openai.api_key = self.config.azure_api_key

            logger.info("OpenAI API設定:")
            logger.info(f"Base URL: {openai.api_base}")
            logger.info(f"API Version: {openai.api_version}")
            logger.info(f"Model: {self.config.azure_model}")

            if not all([openai.api_base, openai.api_key, self.config.azure_model]):
                raise ValueError("必要なOpenAI設定が不足しています")

        except Exception as e:
            logger.error(f"OpenAI API設定エラー: {str(e)}")
            raise

    def _setup_spreadsheet(self):
        """Googleスプレッドシートへの接続を設定"""
        try:
            if self.config.google_credentials_path:
                full_path = os.path.abspath(self.config.google_credentials_path)
                if not os.path.isfile(full_path):
                    raise FileNotFoundError(
                        f"認証情報ファイルが見つかりません: {full_path}"
                    )

            logger.info("Googleスプレッドシートへの接続に成功しました。")
        except Exception as e:
            logger.error(f"Googleスプレッドシート接続エラー: {str(e)}")
            raise

    def generate_personas(
        self, count: int = 5, general: bool = False, use_ai: bool = True
    ) -> List[Persona]:
        """
        ペルソナを生成するメソッド

        Args:
            count: 生成するペルソナの数
            general: 一般的なペルソナを生成するかどうか
            use_ai: AIを使用するかどうか

        Returns:
            List[Persona]: 生成されたペルソナのリスト
        """
        try:
            # AIManager経由でペルソナを生成する
            return self.ai_manager.generate_personas(
                count=count, general=general, use_ai=use_ai
            )
        except Exception as e:
            logger.error(f"ペルソナ生成エラー: {str(e)}")
            raise

        # ペルソナのデータを書き込むコード

    def save_personas_to_sheet(self, personas: List[Persona]) -> None:
        try:
            for persona in personas:
                row_data = [
                    persona.persona_id,
                    persona.age,
                    persona.gender,
                    persona.occupation,
                    # persona.expertise,
                    # persona.experience,
                    persona.thinking_style,
                    persona.work_style,
                    persona.lifestyle,
                    persona.drinking_habits,
                    persona.beer_preferences,
                    persona.recent_concerns,
                    persona.family_composition,
                    persona.information_gathering_methods,
                    persona.health_consciousness,
                    persona.community,
                    persona.criteria_shopping_decisions,
                    persona.leisure_time_usage
                ]
                self.append_row("personas", row_data)
            logger.info(
                f"{len(personas)}件のペルソナ情報をスプレッドシートに保存しました"
            )
        except Exception as e:
            logger.error(f"ペルソナ情報の保存エラー: {str(e)}")
            raise APIError(f"Failed to save personas: {str(e)}")

    # 実行者の名前とIDを保存するコード
    def save_execution_to_sheet(self, execution_id: str, user_name: str) -> None:
        try:
            timestamp = datetime.datetime.now().isoformat()
            row_data = [execution_id, user_name, timestamp]
            self.append_row("executions", row_data)
            logger.info(
                f"実行者情報をスプレッドシートに保存しました: {user_name} ({execution_id})"
            )
        except Exception as e:
            logger.error(f"実行者情報の保存エラー: {str(e)}")
            raise APIError(f"Failed to save execution data: {str(e)}")

    def generate_idea(
        self, persona: Persona, needscope_type: NeedscopeType, idea_prompt: str = ""
    ) -> Optional[Idea]:
        """
        一つのアイディアを生成

        Args:
            persona: ペルソナ情報
            needscope_type: ニードスコープタイプ
            idea_prompt: アイデア生成のためのプロンプト

        Returns:
            Optional[Idea]: 生成されたアイデア
        """
        try:
            return self.ai_manager.generate_idea(
                persona=persona, needscope_type=needscope_type, idea_prompt=idea_prompt
            )
        except Exception as e:
            logger.error(f"アイディア生成エラー: {str(e)}")
            return None

    def save_idea_to_spreadsheet(self, idea: Idea):
        """
        アイデアをスプレッドシートに保存

        Args:
            idea: 保存するアイデア
        """
        try:
            if st.session_state.settings.get("dev_mode", False):
                logger.info(
                    f"開発モード: アイデアの保存をスキップ: {idea.concept_name}"
                )
                return

            self.sheets_client.save_idea(idea)
            logger.info(f"アイデアを保存しました: {idea.concept_name}")

        except Exception as e:
            logger.error(f"アイデア保存エラー: {str(e)}")
            raise

    def analyze_ideas(self) -> Dict[str, Any]:
        """
        生成されたアイデアの分析を実行

        Returns:
            Dict[str, Any]: 分析結果
        """
        try:
            needscope_distribution = self._get_needscope_distribution()
            avg_scores = self._calculate_avg_scores_by_needscope()
            persona_performance = self._analyze_persona_performance()
            keyword_frequency = self._analyze_keyword_frequency()
            time_trends = self._analyze_time_trends()

            return {
                "needscope_distribution": needscope_distribution,
                "avg_scores_by_needscope": avg_scores,
                "persona_performance": persona_performance,
                "keyword_frequency": keyword_frequency,
                "time_trends": time_trends,
                "total_ideas": len(st.session_state.generated_ideas),
            }

        except Exception as e:
            logger.error(f"アイデア分析エラー: {str(e)}")
            return {}


class AIManager:
    """AI関連の操作を管理するクラス"""

    def __init__(self, config: APIConfig):
        self.config = config
        self.openai_client = OpenAIClient(config)
        self.initialize_session_state()
        self.custom_css = self._get_custom_css()

    def _get_custom_css(self) -> str:
        """カスタムCSSを返すメソッド"""
        return """
        .persona-card {
            border: 1px solid #ddd;
            padding: 15px;
            margin: 10px 0;
            border-radius: 5px;
        }
        .persona-header {
            font-weight: bold;
            margin-bottom: 10px;
        }
        """

    def generate_personas(
        self, count: int = 5, general: bool = False, use_ai: bool = True
    ) -> List[Persona]:
        """AIを利用してペルソナを生成するメソッド"""
        personas = []

        system_prompt = """あなたは多様なユーザーペルソナ作成の専門家です。
        以下の形式で、具体的なエピソードと状況を含む詳細なペルソナを生成してください。
        各項目には必ず具体的な説明と、その理由や背景を含めてください。"""

        user_prompt = """
        以下の形式で、詳細なペルソナ情報を生成してください。
        各項目の内容は、具体的なエピソードや状況を含み、箇条書きで記載してください：
        職種は会社員から主婦まで20代以上の社会人としてください。
        
        [以下のjson形式を厳密に守ってください]
        
        {
            "職種": "職種を記載",
            "役職": "役職を記載",
            "年齢": 年齢を数値で記載,
            "性別": "性別を記載",
            "思考スタイル": "内容を記載",
            "仕事の進め方": "内容を記載",
            "ライフスタイル": "内容を記載",
            "飲酒習慣": "内容を記載",
            "ビールの好み": "内容を記載",
            "最近の関心事": "内容を記載",
            "家族構成": "内容を記載",
            "情報の収集方法": "内容を記載",
            "健康の関心毎": "内容を記載",
            "コミュニティ": "内容を記載",
            "買い物時の判断基準": "内容を記載",
            "余暇の時間の使い方": "内容を記載"
        }
        """

        for _ in range(count):
            try:
                if use_ai:
                    response = self.openai_client.generate_completion(
                        [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                        temperature=1,

                    )

                    if isinstance(response, dict) and "choices" in response:
                        content = response["choices"][0]["message"]["content"]
                    else:
                        content = response

                    logger.debug(f"AI Response Content: {content}")

                    persona = self._parse_persona_response(content)
                    if persona:
                        personas.append(persona)
                    else:
                        raise ValueError("AI応答からペルソナを解析できませんでした")
                else:
                    personas.extend(self._generate_random_personas(1, general))

                time.sleep(1)

            except Exception as e:
                logger.error(f"AIによるペルソナ生成エラー: {str(e)}")
                logger.info("フォールバックとしてランダムなペルソナを生成します")
                personas.extend(self._generate_random_personas(1, general))

        return personas

    def _parse_persona_response(self, response: str) -> Optional[Persona]:
        """AIの応答からペルソナオブジェクトを解析"""
        try:
            # JSON形式の応答をパース
            data = json.loads(response)

            # 必須フィールドの存在チェック
            required_fields = ["年齢", "性別", "職種", "役職", "思考スタイル", "仕事の進め方", 
                            "ライフスタイル", "飲酒習慣", "ビールの好み", "最近の関心事",
                            "家族構成", "情報の収集方法", "健康の関心毎", "コミュニティ", 
                            "買い物時の判断基準", "余暇の時間の使い方"]

            if not all(field in data for field in required_fields):
                logger.error("応答に必要なフィールドが欠けています")
                return None

            # Personaオブジェクトを生成し、各フィールドに対応するデータをマッピング
            return Persona(
                persona_id=str(uuid.uuid4()),
                age=int(data["年齢"]),
                gender=data["性別"],
                occupation=f"{data['職種']}、{data['役職']}",
                thinking_style=data["思考スタイル"],
                work_style=data["仕事の進め方"],
                lifestyle=data["ライフスタイル"],
                drinking_habits=data["飲酒習慣"],
                beer_preferences=data["ビールの好み"],
                recent_concerns=data["最近の関心事"],
                family_composition=data["家族構成"],
                information_gathering_methods=data["情報の収集方法"],
                health_consciousness=data["健康の関心毎"],
                community=data["コミュニティ"],
                criteria_shopping_decisions=data["買い物時の判断基準"],
                leisure_time_usage=data["余暇の時間の使い方"]
            )

        except json.JSONDecodeError as e:
            logger.error(f"JSONデコードエラー: {e}")
            return None
        except Exception as e:
            logger.error(f"ペルソナレスポンスのパースエラー: {e}")
            return None
        # try:
        #     # lines = response.strip().split("\n")
        #     current_section = None
        #     data = {
        #         "age": None,
        #         "gender": None,
        #         "occupation": None,
        #         # "expertise": [],
        #         # "experience": [],
        #         "thinking_style": None,
        #         "work_style": None,
        #         "lifestyle": None,
        #         "drinking_habits": None,
        #         "beer_preferences": None,
        #         "recent_concerns":None,
        #         "family_composition":None,
        #         "information_gathering_methods":None,
        #         "health_consciousness":None,
        #         "community":None,
        #         "criteria_shopping_decisions":None,
        #         "leisure_time_usage":None
        #     }

        #     for line in lines:
        #         line = line.strip()
        #         if not line:
        #             continue

        #         if ":" in line and not line.startswith("-"):
        #             header, value = line.split(":", 1)
        #             header = header.strip().lower()

        #             if "ペルソナ" in header:
        #                 match = re.search(r"(.*?)\s*\((\d+)歳\)", value)
        #                 if match:
        #                     data["occupation"] = match.group(1).strip()
        #                     data["age"] = int(match.group(2))
        #             elif "性別" in header:
        #                 data["gender"] = value.strip()

        #             current_section = header

        #         elif line.startswith("-") and current_section:
        #             content = line[1:].strip()
        #             # if "専門分野" in current_section:
        #             #     data["expertise"].append(content)
        #             # elif "経験" in current_section:
        #             #     data["experience"].append(content)
        #             if "思考" in current_section:
        #                 data["thinking_style"].append(content)
        #             elif "仕事" in current_section:
        #                 data["work_style"].append(content)
        #             elif "ライフスタイル" in current_section:
        #                 data["lifestyle"].append(content)
        #             elif "飲酒習慣" in current_section:
        #                 data["drinking_habits"].append(content)
        #             elif "ビール" in current_section:
        #                 data["beer_preferences"].append(content)

        #     for key in [
        #         "expertise",
        #         "experience",
        #         "thinking_style",
        #         "work_style",
        #         "lifestyle",
        #         "drinking_habits",
        #         "beer_preferences",
        #     ]:
        #         if data[key]:
        #             data[key] = "\n".join(f"- {item}" for item in data[key])

        #     if all(
        #         data[field] is not None for field in ["age", "gender", "occupation"]
        #     ):
        #         return Persona(
        #             persona_id=str(uuid.uuid4()),
        #             age=data["age"],
        #             gender=data["gender"],
        #             occupation=data["occupation"],
        #             # expertise=data["expertise"],
        #             # experience=data["experience"],
        #             thinking_style=data["thinking_style"],
        #             work_style=data["work_style"],
        #             lifestyle=data["lifestyle"],
        #             drinking_habits=data["drinking_habits"],
        #             beer_preferences=data["beer_preferences"],
        #         )

        #     return None

        # except Exception as e:
        #     logger.error(f"ペルソナレスポンスのパースエラー: {e}")
        #     return None

    def _generate_random_personas(self, count: int, general: bool = False) -> List[Persona]:
        """フォールバックとしてランダムにペルソナを生成"""
        personas = []
        occupations = [
            {"職種": "商品開発", "役職": "商品開発マネージャー"},
            {"職種": "マーケティング", "役職": "マーケティング責任者"},
            {"職種": "品質管理", "役職": "品質管理スペシャリスト"},
            {"職種": "営業", "役職": "営業統括部長"},
            {"職種": "消費者調査", "役職": "消費者調査アナリスト"},
        ]

        for i in range(count):
            occupation = occupations[i % len(occupations)]
            personas.append(
                Persona(
                    persona_id=str(uuid.uuid4()),
                    occupation=occupation["職種"],
                    # job_title=occupation["役職"],
                    age=random.randint(30, 60),
                    gender=random.choice(["男性", "女性"]),
                    thinking_style=random.choice(["クリエイティブ", "分析的"]),
                    work_style=random.choice(["チーム協調", "個人プレー"]),
                    lifestyle=random.choice(["アクティブ", "落ち着いた"]),
                    drinking_habits=random.choice(["週末に飲む", "日常的に飲む"]),
                    beer_preferences=random.choice(
                        ["クラフトビール", "レギュラービール", "ノンアルコールビール"]
                    ),
                    recent_concerns=random.choice(["健康", "仕事の効率","趣味","アイドル","ゲーム"]),
                    family_composition=random.choice(["独身", "配偶者あり", "配偶者と子供"]),
                    information_gathering_methods=random.choice(["SNS", "インターネット検索", "テレビ","新聞"]),
                    health_consciousness=random.choice(["高い", "中程度", "低い"]),
                    community=random.choice(["地域サークル", "職場コミュニティ", "オンラインフォーラム"]),
                    criteria_shopping_decisions=random.choice(["品質", "価格", "ブランド"]),
                    leisure_time_usage=random.choice(["アウトドア活動", "読書", "趣味活動"])
                )
            )
        return personas

    def _get_custom_css(self) -> str:
        """カスタムCSSの定義を返す"""
        # 既存の_get_custom_cssメソッドの内容をそのまま維持
        return """
        <style>
            /* 既存のCSSスタイル定義 */
        </style>
        """

    def initialize_session_state(self):
        """セッション状態の初期化"""
        if "settings" not in st.session_state:
            st.session_state.settings = {
                "batch_size": 3,
                "auto_save": True,
                "dev_mode": False,
            }
        if "current_session_id" not in st.session_state:
            st.session_state.current_session_id = str(uuid.uuid4())
        if "generated_ideas" not in st.session_state:
            st.session_state.generated_ideas = []
        if "personas" not in st.session_state:
            st.session_state.personas = []
        if "personas_confirmed" not in st.session_state:
            st.session_state.personas_confirmed = False
        if "current_batch_index" not in st.session_state:
            st.session_state.current_batch_index = 0
        if "user_info" not in st.session_state:
            st.session_state.user_info = None

    def generate_idea(
        self, persona: Persona, needscope_type: NeedscopeType, idea_prompt: str = ""
    ) -> Optional[Idea]:
        """
        ペルソナとニードスコープに基づいてビールコンセプトのアイデアを生成

        Args:
            persona: ペルソナ情報
            needscope_type: ニードスコープタイプ
            idea_prompt: 追加のプロンプト（任意）

        Returns:
            Optional[Idea]: 生成されたアイデア、失敗時はNone
        """
        try:
            # プロンプトの構築
            prompt = self._create_idea_generation_prompt(
                persona, needscope_type, idea_prompt
            )

            # OpenAI APIを使用してアイデアを生成
            response = self.openai_client.generate_completion(
                [
                    {
                        "role": "system",
                        "content": "あなたは創造的なビール商品開発の専門家です。",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.8,
            )

            # レスポンスをパース
            idea = self._parse_idea_response(response, persona, needscope_type)
            if idea:
                return idea

        except Exception as e:
            logger.error(f"アイディア生成エラー: {str(e)}")
            return None

    def _create_idea_generation_prompt(
        self, persona: Persona, needscope_type: NeedscopeType, idea_prompt: str
    ) -> str:
        """アイデア生成用のプロンプトを作成"""
        needscope_info = NEEDSCOPE_DATA[needscope_type]
        
        prompt = f"""
            以下のペルソナとニードスコープに基づいて、ビールのコンセプトを生成してください。

            【背景】
            ビール会社が新しい商品を企画しており、市場に価値をもたらす革新的な新商品・サービスのアイデアを発想したいと考えています。
            
            【コンセプトの方針】
            {idea_prompt}

            【ターゲットペルソナ】
            年齢: {persona.age}
            性別: {persona.gender}
            職業: {persona.occupation}
            ライフスタイル: {persona.lifestyle}
            飲酒習慣: {persona.drinking_habits}
            ビール好み: {persona.beer_preferences}
            最近の懸念: {persona.recent_concerns}
            家族構成: {persona.family_composition}
            情報収集方法: {persona.information_gathering_methods}
            健康意識: {persona.health_consciousness}
            コミュニティ: {persona.community}
            購買決定基準: {persona.criteria_shopping_decisions}
            余暇時間の使い方: {persona.leisure_time_usage}

            【ニードスコープ】
            タイプ: {needscope_info['name']}
            特徴: {needscope_info['description']}
            キーワード: {', '.join(needscope_info['keywords'])}
            得られる感情：{needscope_info['emotional_value']}
            
            ###jsonの形式
            {{
                "製品名":"商品の名前",
                "コンセプト":"商品のコンセプト",
                "特徴":"ビールのパッケージや中身の特徴",
                "想定価格":"値段（例：100円）",
                "タグライン":"ビールの説明",
                "accepted_consumer_belief":"ユーザーの困りごと",
                "reason_to_believe":"顧客にとってお困りを解決するに値する理由（主に中身やパッケージとったスペック）"
            }}

            【トーンとスタイル】
            - イノベーティブで創造的な表現を心がける
            - 論理的で説得力のある文章を心がける
            - 明確かつ簡潔に伝える

            ###条件（必ず守ること）
            ・コンセプト:商品のコンセプトは100～140字程度で、価値やユーザー、ターゲット、商品特徴を含めてください。
            ・タグラインは30文字程度
            ・Accepted Consumer Beliefは40文字程度
            ・reason to believeは100～120文字

        """
        # prompt = f"""
        #     以下のペルソナとニードスコープに基づいて、独創的なビールのコンセプトを生成してください。

        #     [ペルソナ情報]
        #     - 年齢: {persona.age}歳
        #     - 職種: {persona.occupation}
        #     - ビール好み: {persona.beer_preferences}

        #     [ニードスコープ]
        #     - タイプ: {needscope_info['name']}
        #     - キーワード: {', '.join(needscope_info['keywords'])}
        #     - 感情価値: {needscope_info['emotional_value']}
        #     - 消費機会: {needscope_info['consumption_occasion']}

        #     [追加の要望]
        #     {idea_prompt}

        #     以下の形式で出力してください:
        #     商品名：[商品名]
        #     コンセプト：[150-250文字程度でコンセプトを説明]

        # """

        return prompt
    def _parse_idea_response(
        self, response: str, persona: Persona, needscope_type: NeedscopeType
    ) -> Optional[Idea]:
        """AIのJSON応答をアイデアオブジェクトにパース"""
        try:
            # JSON形式の応答をパース
            data = json.loads(response)

            # 必須フィールドの存在確認
            required_fields = ["製品名", "コンセプト", "特徴", "想定価格", "タグライン", 
                            "accepted_consumer_belief", "reason_to_believe"]

            if not all(field in data for field in required_fields):
                logger.error("応答に必要なフィールドが欠けています")
                return None

            # ニードスコープ分析を実行
            needscope_analysis = analyze_text_for_needscope(data["コンセプト"])
            evaluation_score = sum(analysis.total_score for analysis in needscope_analysis)

            # Ideaオブジェクトを生成
            return Idea(
                idea_id=str(uuid.uuid4()),
                session_id=st.session_state.current_session_id,
                persona_id=persona.persona_id,
                needscope_type=needscope_type,
                concept_name=data["製品名"],
                description=data["コンセプト"],
                features=data["特徴"],
                price=data["想定価格"],
                tagline=data["タグライン"],
                accepted_consumer_belief=data["accepted_consumer_belief"],
                reason_to_believe=data["reason_to_believe"],
                evaluation_score=evaluation_score,
                created_at=datetime.datetime.now(),
            )

        except json.JSONDecodeError as e:
            logger.error(f"JSONデコードエラー: {e}")
            return None
        except Exception as e:
            logger.error(f"アイデアレスポンスのパースエラー: {str(e)}")
            return None
    
    # def _parse_idea_response(
    #     self, response: str, persona: Persona, needscope_type: NeedscopeType
    # ) -> Optional[Idea]:
    #     """AIの応答をアイデアオブジェクトにパース"""
    #     try:
    #         lines = response.strip().split("\n")
    #         concept_name = ""
    #         description = ""

    #         for line in lines:
    #             if "商品名：" in line:
    #                 concept_name = line.split("：", 1)[1].strip()
    #             elif "コンセプト：" in line:
    #                 description = line.split("：", 1)[1].strip()

    #         if concept_name and description:
    #             # ニードスコープ分析を実行
    #             needscope_analysis = analyze_text_for_needscope(description)
    #             evaluation_score = sum(
    #                 analysis.total_score for analysis in needscope_analysis
    #             )

    #             return Idea(
    #                 idea_id=str(uuid.uuid4()),
    #                 session_id=st.session_state.current_session_id,
    #                 persona_id=persona.persona_id,
    #                 needscope_type=needscope_type,
    #                 concept_name=concept_name,
    #                 description=description,
    #                 evaluation_score=evaluation_score,
    #                 created_at=datetime.datetime.now(),
    #             )

    #     except Exception as e:
    #         logger.error(f"アイデアレスポンスのパースエラー: {str(e)}")
    #         return None


if TYPE_CHECKING:
    from .core import BeerStormSystem


class UIManager:
    def __init__(self, system: "BeerStormSystem"):
        self.system = system
        self.initialize_session_state()
        self.custom_css = self._get_custom_css()

    def initialize_session_state(self):
        """セッション状態の初期化"""
        if "settings" not in st.session_state:
            st.session_state.settings = {
                "batch_size": 3,
                "auto_save": True,
                "dev_mode": False,
            }
        if "current_session_id" not in st.session_state:
            st.session_state.current_session_id = str(uuid.uuid4())
        if "generated_ideas" not in st.session_state:
            st.session_state.generated_ideas = []
        if "personas" not in st.session_state:
            st.session_state.personas = []
        if "personas_confirmed" not in st.session_state:
            st.session_state.personas_confirmed = False
        if "current_batch_index" not in st.session_state:
            st.session_state.current_batch_index = 0
        if "user_info" not in st.session_state:
            st.session_state.user_info = None

    def setup_page(self):
        st.set_page_config(page_title="BeerStormApp", layout="wide")
        st.title("BeerStormApp")
        st.write("AIを活用して新しいビールコンセプトを生成し、詳細な分析を行います。")

    def render_sidebar(self) -> str:
        st.sidebar.title("メニュー")
        return st.sidebar.radio(
            "選択してください",
            ("ホーム", "アイディア生成", "セッション管理", "設定"),
        )

    def render_home(self):
        """ホーム画面の表示"""
        st.markdown("### 🍺 Beer Storm")

        if not st.session_state.user_info:
            self._render_user_registration()
        else:
            # ユーザー情報とセッション情報の表示
            st.sidebar.success(f"👤 {st.session_state.user_info['name']}")
            if "current_session_id" in st.session_state:
                st.sidebar.info(
                    f"🔑 現在のセッションID: {st.session_state.current_session_id}"
                )

            if not st.session_state.personas:
                st.markdown("#### まずは企画チームを生成しましょう！")
                col1, col2 = st.columns([3, 1])
                with col1:
                    use_ai = st.checkbox("AIを使用してペルソナを生成", value=True)
                with col2:
                    team_size = st.number_input(
                        "チームサイズ", min_value=3, max_value=7, value=5
                    )

                if st.button("企画チームを生成", use_container_width=True):
                    with st.spinner("企画チームを生成中..."):
                        # 新しいセッションを作成
                        session_id = self.system.spreadsheet_db.create_session(
                            st.session_state.user_info["user_id"],
                            {"team_size": team_size, "use_ai": use_ai},
                        )
                        st.session_state.current_session_id = session_id

                        # ペルソナを生成
                        st.session_state.personas = self.system.generate_personas(
                            count=team_size, general=False, use_ai=use_ai
                        )
                        st.success(
                            f"新しいセッションが開始されました！\nセッションID: {session_id}"
                        )
                        # st.rerun()
            else:
                self._render_persona_editor()

    def _render_persona_editor(self):
        """ペルソナ編集画面の表示"""
        st.markdown("### 企画チームの編集")

        # すでに生成されたペルソナが存在する場合、それぞれのペルソナを表示
        if st.session_state.personas:
            for persona in st.session_state.personas:
                with st.expander(f"ペルソナ: {persona.occupation} ({persona.age}歳)"):
                    st.write(f"性別: {persona.gender}")
                    # st.write(f"専門分野: {persona.expertise}")
                    # st.write(f"経験: {persona.experience}")
                    st.write(f"思考スタイル: {persona.thinking_style}")
                    st.write(f"仕事の進め方: {persona.work_style}")
                    st.write(f"ライフスタイル: {persona.lifestyle}")
                    st.write(f"飲酒習慣: {persona.drinking_habits}")
                    st.write(f"ビールの好み: {persona.beer_preferences}")
                    st.write(f"最近の懸念: {persona.recent_concerns}")
                    st.write(f"家族構成: {persona.family_composition}")
                    st.write(f"情報収集方法: {persona.information_gathering_methods}")
                    st.write(f"健康意識: {persona.health_consciousness}")
                    st.write(f"コミュニティ: {persona.community}")
                    st.write(f"購買決定基準: {persona.criteria_shopping_decisions}")

            if st.button("ペルソナを確定", key="confirm_personas"):
                st.session_state.personas_confirmed = True
                st.success("ペルソナが確定されました。アイディア生成に進めます。")
        else:
            st.warning("現在、ペルソナが存在しません。ホーム画面で生成してください。")

    def render_idea_generation(self):
        """アイディア生成画面の表示"""
        if not st.session_state.personas_confirmed:
            st.warning("まずはホーム画面で企画チームを確定させてください。")
            return

        st.markdown("### アイデア生成")

        # 生成設定のセクション
        with st.expander("生成設定", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                batch_size = st.number_input(
                    "バッチサイズ",
                    min_value=1,
                    max_value=20,
                    value=10,  # デフォルトを10に設定
                )
                idea_prompt = st.text_input(
                    "生成したいアイディアのテーマを入力してください",
                    placeholder="例: 新しいビール体験",
                )

            with col2:
                needscope_selection = st.multiselect(
                    "ニードスコープを選択してください",
                    options=list(NeedscopeType),
                    format_func=lambda x: NEEDSCOPE_DATA[x]["name"],
                )

        # 進捗状況の表示
        total_ideas = len(st.session_state.generated_ideas)
        st.progress(total_ideas / idea_size)
        st.markdown(f"**総生成数**: {total_ideas}/{idea_size}")

        # バッチ生成のコントロール
        if "is_generating" not in st.session_state:
            st.session_state.is_generating = False

        col1, col2 = st.columns([2, 1])
        with col1:
            button_text = (
                "自動生成を停止" if st.session_state.is_generating else "自動生成を開始"
            )
            if st.button(button_text):
                st.session_state.is_generating = not st.session_state.is_generating
                # st.rerun()

        with col2:
            if st.button("一時保存", disabled=st.session_state.is_generating):
                self._save_current_batch()
                st.success("現在のバッチを保存しました")

        # 自動生成モードの実行
        if st.session_state.is_generating and total_ideas < idea_size:
            current_batch = []
            progress_bar = st.progress(0)
            status_text = st.empty()

            try:
                # バッチサイズ分のアイデアを生成
                for i in range(batch_size):
                    if total_ideas + i >= idea_size:
                        break

                    # ランダムにニードスコープを選択（選択されていない場合）
                    if not needscope_selection:
                        needscope = random.choice(list(NeedscopeType))
                    else:
                        needscope = random.choice(needscope_selection)

                    # ランダムにペルソナを選択
                    persona = random.choice(st.session_state.personas)

                    status_text.text(f"アイデア {i+1}/{batch_size} を生成中...")
                    progress_bar.progress((i + 1) / batch_size)

                    idea = self.system.ai_manager.generate_idea(
                        persona=persona,
                        needscope_type=needscope,
                        idea_prompt=idea_prompt,
                    )

                    if idea:
                        current_batch.append(idea)
                        self._display_idea_card(idea)
                        st.session_state.generated_ideas.append(idea)
                        time.sleep(0.5)  # API制限を考慮
                        print(idea)

                # バッチ完了後に自動保存
                if current_batch:
                    status_text.text("バッチを保存中...")
                    self._save_batch_to_spreadsheet(current_batch)
                    status_text.text("✅ バッチの生成と保存が完了しました")
                    time.sleep(2)
                    st.rerun()  # 次のバッチのために画面を更新

            except Exception as e:
                st.error(f"エラーが発生しました: {str(e)}")
                st.session_state.is_generating = False
                logger.error(f"バッチ生成エラー: {str(e)}")

            if total_ideas >= idea_size:
                st.success(f"🎉 {idea_size}個のアイデア生成が完了しました！")
                st.session_state.is_generating = False

    def _display_idea_card(self, idea: Idea):
        """アイデアカードの表示"""
        needscope_info = NEEDSCOPE_DATA[idea.needscope_type]

        # カードスタイルの定義
        card_style = f"""
        <style>
            .idea-card {{
                background-color: #1E1E1E;
                padding: 1.5rem;
                border-radius: 8px;
                margin-bottom: 1rem;
                border: 1px solid #333;
            }}
            .needscope-badge {{
                background-color: {needscope_info['color']};
                color: white;
                padding: 4px 12px;
                border-radius: 15px;
                font-size: 0.875rem;
                font-weight: 500;
                display: inline-block;
                margin-bottom: 0.5rem;
            }}
            .idea-title {{
                color: white;
                font-size: 1.2rem;
                margin: 0.5rem 0;
            }}
            .idea-description {{
                color: #CCC;
                margin-bottom: 1rem;
            }}
            .idea-meta {{
                color: #888;
                font-size: 0.875rem;
                display: flex;
                justify-content: space-between;
            }}
        </style>
        """

        # カードのHTML
        card_html = f"""
        <div class="idea-card">
            <span class="needscope-badge">
                {needscope_info['name']}
            </span>
            <h3 class="idea-title">{idea.concept_name}</h3>
            <p class="idea-description">{idea.description}</p>
            <div class="idea-meta">
                <span>評価スコア: {idea.evaluation_score:.1f}</span>
                <span>{idea.created_at.strftime('%Y-%m-%d %H:%M')}</span>
            </div>
        </div>
        """

        st.markdown(card_style + card_html, unsafe_allow_html=True)

    def _save_batch_to_spreadsheet(self, batch: List[Idea]):
        """バッチをスプレッドシートに保存"""
        try:
            for idea in batch:
                # NeedscopeTypeをstrに変換
                needscope_value = idea.needscope_type.value if idea.needscope_type else ""

                # 日時をISO形式の文字列に変換
                created_at_str = idea.created_at.isoformat() if idea.created_at else ""

                # 保存するデータの準備
                row_data = [
                    str(idea.idea_id),
                    str(idea.session_id),
                    str(idea.persona_id),
                    needscope_value,
                    str(idea.concept_name),
                    str(idea.description),
                    str(idea.features),
                    str(idea.price),
                    str(idea.tagline),
                    str(idea.accepted_consumer_belief),
                    str(idea.reason_to_believe),
                    str(idea.evaluation_score),
                    created_at_str,
                ]

                try:
                    self.system.sheets_client.append_row("ideas", row_data)
                    logger.info(f"アイデアを保存: {idea.concept_name}")
                    time.sleep(0.5)  # APIレート制限を考慮
                except Exception as e:
                    logger.error(f"個別アイデアの保存エラー: {str(e)}")
                    continue  # 1件の失敗で全体を止めない

            logger.info(f"バッチ {len(batch)} 件の保存を完了")

        except Exception as e:
            logger.error(f"バッチ保存処理エラー: {str(e)}")
            st.error("データの保存中にエラーが発生しました。")
            raise

    # def _save_batch_to_spreadsheet(self, batch: List[Idea]):
    #     """バッチをスプレッドシートに保存"""
    #     try:
    #         for idea in batch:
    #             # NeedscopeTypeをstrに変換
    #             needscope_value = (
    #                 idea.needscope_type.value if idea.needscope_type else ""
    #             )

    #             # 日時をISO形式の文字列に変換
    #             created_at_str = idea.created_at.isoformat() if idea.created_at else ""

    #             # 保存するデータの準備
    #             row_data = [
    #                 str(idea.idea_id),
    #                 str(idea.session_id),
    #                 str(idea.persona_id),
    #                 needscope_value,
    #                 str(idea.concept_name),
    #                 str(idea.description),
    #                 str(idea.evaluation_score),
    #                 created_at_str,
    #             ]

    #             try:
    #                 self.system.sheets_client.append_row("ideas", row_data)
    #                 logger.info(f"アイデアを保存: {idea.concept_name}")
    #                 time.sleep(0.5)  # APIレート制限を考慮
    #             except Exception as e:
    #                 logger.error(f"個別アイデアの保存エラー: {str(e)}")
    #                 continue  # 1件の失敗で全体を止めない

    #         logger.info(f"バッチ {len(batch)} 件の保存を完了")

    #     except Exception as e:
    #         logger.error(f"バッチ保存処理エラー: {str(e)}")
    #         st.error("データの保存中にエラーが発生しました。")
    #         raise

    def _render_user_registration(self):
        st.markdown("### 👤 ユーザー名を入力してください")
        with st.form("user_registration"):
            name = st.text_input("お名前")

            if st.form_submit_button("開始"):
                if name:
                    user_id = str(uuid.uuid4())  # 単純にUUIDを生成
                    st.session_state.user_info = {
                        "user_id": user_id,
                        "name": name,
                    }
                    st.success("ユーザー情報を登録しました！")
                    st.rerun()
                else:
                    st.error("名前を入力してください。")

    def _get_custom_css(self) -> str:
        """カスタムCSSの定義を返す"""
        return """
            <style>
            .beer-storm-header {
                background: linear-gradient(135deg, #F39C12 0%, #F1C40F 100%);
                padding: 2rem;
                border-radius: 10px;
                color: white;
                margin-bottom: 2rem;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            }

            .idea-card {
                background-color: #1E1E1E;
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

            .needscope-badge {
                color: white;
                padding: 4px 12px;
                border-radius: 15px;
                font-size: 0.875rem;
                font-weight: 500;
                display: inline-block;
                margin-bottom: 0.5rem;
            }

            .idea-meta {
                margin-top: 1rem;
                display: flex;
                justify-content: space-between;
                color: #666;
                font-size: 0.875rem;
            }
            </style>
        """

    def render_session_management(self):
        """セッション管理ページの表示"""
        st.markdown("### セッション管理")

        session_id_input = st.text_input("セッションIDを入力してください", "")

        if st.button("セッション結果を表示"):
            if session_id_input:
                try:
                    # セッションIDを元に結果を取得
                    logger.info(
                        f"Retrieving results for session ID: {session_id_input}"
                    )
                    results = self.system.spreadsheet_db.get_session_results(
                        session_id_input
                    )

                    # デバッグ用に取得結果を表示
                    logger.info(f"Retrieved results: {results}")
                    st.write("取得結果:", results)

                    # 結果が存在する場合
                    if results["session"]:
                        # セッション情報の表示
                        st.markdown("#### セッション情報")
                        session = results["session"]
                        st.write(
                            {
                                "セッションID": session.session_id,
                                "ユーザーID": session.user_id,
                                "開始時間": session.start_time,
                                "終了時間": session.end_time,
                                "状態": session.status,
                                "アイデア総数": session.total_ideas,
                            }
                        )

                        # アイデアリストの表示
                        st.markdown("#### 生成されたアイデア")
                        for idea in results["ideas"]:
                            st.markdown(f"**アイデアID**: {idea.idea_id}")
                            st.markdown(f"**コンセプト名**: {idea.concept_name}")
                            st.markdown(f"**評価スコア**: {idea.evaluation_score:.2f}")
                            st.markdown(
                                f"**生成日時**: {idea.created_at.strftime('%Y-%m-%d %H:%M')}"
                            )
                            st.markdown(f"**説明**: {idea.description}")
                            st.divider()
                    else:
                        st.warning(
                            "指定されたセッションIDの結果が見つかりませんでした。"
                        )
                except Exception as e:
                    logger.error(f"Error retrieving session results: {e}")
                    st.error(f"エラーが発生しました: {e}")
            else:
                st.warning("セッションIDを入力してください。")

    def render_analysis_section(self):
        """分析画面のレンダリング"""
        st.subheader("アイディア分析")
        analysis_data = self.system.analyze_ideas()
        if analysis_data:
            st.write("ニードスコープ分布:")
            st.write(analysis_data["needscope_distribution"])
            st.write("キーワード頻度分析:")
            st.write(analysis_data["keyword_frequency"])
        else:
            st.warning("分析データがまだありません。")

    def render_settings(self):
        """設定画面のレンダリング"""
        st.subheader("アプリ設定")
        st.write("ここでアプリの設定を変更します。")
        dev_mode = st.checkbox(
            "開発モード", value=st.session_state.settings["dev_mode"]
        )
        st.session_state.settings["dev_mode"] = dev_mode


def run():
    """メインアプリケーション"""
    try:
    
        # # ページ設定を最初に行う
        # st.set_page_config(page_title="BeerStormApp", layout="wide")
        # logger.info("ページをセットアップしました。")

        # 環境変数の読み込み
        load_dotenv()
        logger.info("環境変数を読み込みました。")

        # ベースディレクトリの取得（このファイルの親の親ディレクトリ）
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        # クレデンシャルパスの解決
        google_credentials_path = os.getenv("GOOGLE_CREDENTIALS_PATH")
        credentials_filename = os.path.basename(google_credentials_path)
        absolute_credentials_path = os.path.join(base_dir, credentials_filename)

        logger.info(f"Base directory: {base_dir}")
        logger.info(f"Looking for credentials file: {credentials_filename}")
        logger.info(f"Absolute credentials path: {absolute_credentials_path}")

        if not os.path.exists(absolute_credentials_path):
            raise FileNotFoundError(
                f"Google credentials file not found at: {absolute_credentials_path}"
            )

        # 環境変数を絶対パスで更新
        os.environ["GOOGLE_CREDENTIALS_PATH"] = absolute_credentials_path
        logger.info(f"Using credentials at: {absolute_credentials_path}")

        # セッション状態の初期化
        init_session_state()
        logger.info("セッション状態を初期化しました。")

        # API設定の初期化
        api_config = APIConfig.from_env()
        logger.info("API設定を読み込みました。")

        # システムとUIの初期化
        system = BeerStormSystem(api_config)
        ui_manager = UIManager(system)
        logger.info("システムとUIの初期化が完了しました。")

        # サイドバーのメニュー表示と選択内容を取得
        selected_menu = ui_manager.render_sidebar()
        logger.info(f"選択されたメニュー: {selected_menu}")

        # 選択されたメニューに応じて画面を表示
        if selected_menu == "ホーム":
            ui_manager.render_home()
            logger.info("ホーム画面を表示しました。")
        elif selected_menu == "アイディア生成":
            ui_manager.render_idea_generation()
            logger.info("アイディア生成画面を表示しました。")
        elif selected_menu == "セッション管理":
            ui_manager.render_session_management()
            logger.info("セッション管理画面を表示しました。")
        elif selected_menu == "設定":
            ui_manager.render_settings()
            logger.info("設定画面を表示しました。")
        else:
            st.write("メニューの選択に誤りがあります。")
            logger.warning(f"不明なメニューが選択されました: {selected_menu}")

    except Exception as e:
        logger.error(f"Application error: {str(e)}")
        st.error(f"アプリケーションエラーが発生しました: {str(e)}")

if __name__ == "__main__":
    run()