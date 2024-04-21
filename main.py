from sqlalchemy.orm import Session
from fastapi import FastAPI, Depends, HTTPException, Query, APIRouter, Header
from fastapi.middleware.cors import CORSMiddleware  # CORSMiddlewareをインポート
from pydantic import BaseModel
from sqlalchemy import Boolean, create_engine, Column, Integer, String, DateTime, Numeric, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.orm import relationship
from datetime import datetime, timedelta
from fastapi.responses import JSONResponse
import os
from typing import Optional, List
from sqlalchemy import func
from jose import jwt, JWTError
import pytz
from passlib.context import CryptContext
import pandas as pd

# MongoDB接続ライブラリ
from pymongo import MongoClient
from typing import List, Optional
from bson import ObjectId

# OpenAI接続
from openai import OpenAI
import numpy as np
from scipy.spatial.distance import cosine

# 環境変数の読み込み
from dotenv import load_dotenv
load_dotenv()

DATABASE_URL = os.getenv('DATABASE_URL')
if not DATABASE_URL:
    raise EnvironmentError("DATABASE_URL is not set in .env file")

SECRET_KEY = os.getenv('SECRET_KEY')
if not SECRET_KEY:
    raise EnvironmentError("SECRET_KEY is not set in .env file")

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
if not OPENAI_API_KEY:
    raise EnvironmentError("OPENAI_API_KEY is not set in .env file")

MONGODB_URL = os.getenv('MONGODB_URL')
if not DATABASE_URL:
    raise EnvironmentError("MONGODB_URL is not set in .env file")

ALGORITHM = "HS256"
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

app = FastAPI()

# CORS設定
app.add_middleware(
    CORSMiddleware,
        allow_origins=[
        "https://tech0-gen-5-step4-studentwebapp-1.azurewebsites.net",
        "https://tech0-gen-5-step4-studentwebapp-2.azurewebsites.net"
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],  # 許可するメソッドを指定します。
    allow_headers=["*"],  # すべてのヘッダーを許可します。必要に応じて指定します。
)

# データベース接続設定
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class SurveyResponseCreate(BaseModel):
    question_id: str
    answer_text: str

class SurveyResponse(Base):
    __tablename__ = 'survey_responses'
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey('users.user_id'))
    question_id = Column(String, nullable=False)  # 質問IDを格納するカラム（文字列型）
    answer_text = Column(String, nullable=False)  # 回答テキスト
    created_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("UserDB", back_populates="survey_responses")

class SurveyResponseSchema(BaseModel):
    id: int
    user_id: int
    answer_text: str
    created_at: datetime

    class Config:
        from_attributes = True

class ProductDB(Base):
    __tablename__ = 'products'
    product_id = Column(Integer, primary_key=True, nullable=False)
    product_name = Column(String(255), nullable=False)
    manufacturer = Column(String(255), nullable=False)
    volume = Column(String(255), nullable=False)
    fragrance = Column(String(255), nullable=False)
    color = Column(String(255), nullable=False)
    category = Column(String(255), nullable=False)
    excluding_tax_price = Column(Integer, nullable=False)
    including_tax_price = Column(Integer, nullable=False)
    product_qrcode = Column(Integer, unique=True, nullable=False)
    quantity = Column(Integer, nullable=False)
    last_update = Column(DateTime, default=datetime.utcnow)
    image_url = Column(String(255), nullable=True)  # 画像URLを保存するカラムを追加

class UserDB(Base):
    __tablename__ = 'users'
    user_id = Column(Integer, primary_key=True, nullable=False)
    user_name = Column(String(255), nullable=False)
    hashed_password = Column(String(1000), nullable=True)
    token = Column(String(1000), nullable=True)
    last_update = Column(DateTime, default=datetime.utcnow)
    first_login = Column(Boolean, default=True)
    survey_responses = relationship("SurveyResponse", back_populates="user")

# ユーザー情報を格納するためのモデル
class User(BaseModel):
    user_name: str
    password: str

# ユーザー登録用のリクエストボディモデル
class UserCreate(BaseModel):
    user_name: str
    password: str

class UserResponse(BaseModel):
    user_id: int
    user_name: str
    # password フィールドは含まない
    class Config:
        from_attributes = True  # ORMモデルとの互換性のために追加


# データベースのセットアップ
Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

# ユーザー認証を行う関数
def get_current_user(db: Session = Depends(get_db), token:str = Query(...)):
    try:
        # トークンのデコード
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(status_code=401, detail="Invalid JWT token")
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid JWT token")

    # データベースからユーザーを取得
    user = db.query(UserDB).filter(UserDB.user_name == username).first()
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")
    return user

class Purchase_HistoryDB(Base):
    __tablename__ = 'purchase_history'  # テーブル名を指定
    purchase_id = Column(Integer, primary_key=True, nullable=False)
    user_id = Column(Integer, nullable=True)  # MySQL定義に合わせてデフォルトNULL
    product_id = Column(Integer, nullable=True)  # MySQL定義に合わせてデフォルトNULL
    quantity = Column(Integer, nullable=False)
    favorite = Column(Boolean, nullable=False, default=False)
    registration_date = Column(DateTime, nullable=False)  # nullable=TrueからFalseに変更


# リクエストボディのモデル定義
class Product(BaseModel):
    product_qrcode: int
    product_name: str
    price: int
    quantity: int
    tax_percent: float
    buy_time: datetime

class ProductList(BaseModel):
    products: List[Product]

# リクエストデータモデル
class PurchaseHistoryCreate(BaseModel):
    user_id: int
    product_id: int
    quantity: int
    favorite: bool
    registration_date: datetime

# レスポンスデータモデル
class PurchaseHistoryResponse(BaseModel):
    purchase_id: int
    user_id: int
    product_id: int
    quantity: int
    favorite: bool
    registration_date: datetime

    class Config:
        from_attributes = True

# 購入履歴のレスポンスモデルを定義
class RecentPurchase(BaseModel):
    purchase_id: int
    product_name: str
    quantity: int
    registration_date: datetime
    image_url: str

class FavoriteProduct(BaseModel):
    product_id: int
    product_name: str
    including_tax_price: int

class MyPageResponse(BaseModel):
    recent_purchases: List[RecentPurchase]
    favorite_products: List[FavoriteProduct]

# 商品情報登録用のPydanticモデル
class PurchaseHistoryCreate(BaseModel):
    product_id: int
    quantity: int
    favorite: bool
    registration_date: datetime

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

# UTCで取得した日時をJSTに変換する関数
def to_jst(datetime_obj):
    utc_zone = pytz.utc
    jst_zone = pytz.timezone('Asia/Tokyo')
    return datetime_obj.replace(tzinfo=utc_zone).astimezone(jst_zone)

router = APIRouter()

# 認証用ユーティリティ関数
def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def get_user_by_token(db: Session, token: str):
    return db.query(UserDB).filter(UserDB.token == token).first()

@app.post("/register", response_model=UserResponse)
async def create_user(user: UserCreate, db: Session = Depends(get_db)):
    # データベースに既に同じユーザー名が存在するかチェック
    db_user = db.query(UserDB).filter(UserDB.user_name == user.user_name).first()
    if db_user:
        raise HTTPException(status_code=400, detail="Username already registered")

    # パスワードをハッシュ化
    hashed_password = get_password_hash(user.password)

    # 新しいユーザーオブジェクトを作成
    db_user = UserDB(user_name=user.user_name, hashed_password=hashed_password)

    # 新しいユーザーをデータベースに追加した後
    db.add(db_user)
    db.commit()
    db.refresh(db_user)

    # レスポンス用に UserResponse モデルを使用
    return UserResponse(user_id=db_user.user_id, user_name=db_user.user_name)

@app.post('/login')
async def login(user: User, db: Session = Depends(get_db)):
    user_info = db.query(UserDB).filter(UserDB.user_name == user.user_name).first()
    if not user_info:
        raise HTTPException(status_code=401, detail="Incorrect username or password")

    if not verify_password(user.password, user_info.hashed_password):
        raise HTTPException(status_code=401, detail="Incorrect username or password")

    # 初回ログインチェック
    if user_info.first_login:
        # トークン生成と初回フラグの更新
        user_info.first_login = False  # フラグを更新
        db.commit()

    now = datetime.now()
    # トークンの更新または新規作成の必要性をチェック
    if user_info.token is None or (now - user_info.last_update) > timedelta(days=7):
        # トークンのペイロード
        payload = {
            "sub": user_info.user_name,
            "exp": now + timedelta(days=7)  # トークンの有効期限を7日に設定
        }
        # トークンを生成
        access_token = jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)
        # ユーザー情報の更新
        user_info.token = access_token
        user_info.last_update = now
        db.commit()  # データベースに変更を保存
    else:
        access_token = user_info.token

    return {"access_token": access_token, "user_name": user_info.user_name, "first_login": not user_info.first_login}

@app.get('/shopping')
async def read_login(token: str = Query(..., description="Token information")):
    db = SessionLocal()
    # ユーザーの認証
    user_info = db.query(UserDB).filter_by(token=token).first()
    if not user_info:
        raise HTTPException(status_code=401, detail="Bad token")

    user_name = user_info.user_name

    # ユーザー名とユーザーIDをクライアントに返す
    return {"user_name": user_name, "user_id": user_info.user_id}

@app.get('/mypage', response_model=MyPageResponse)
async def read_user_data(token: str = Query(..., description="Token information"), db: Session = Depends(get_db)):
    # トークンを使用してユーザー情報を認証し取得 ...
    user_info = db.query(UserDB).filter_by(token=token).first()
    if not user_info:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

    user_id = user_info.user_id

    # 直近10件の購入履歴とお気に入り商品のIDを取得
    recent_purchases_query = db.query(
        Purchase_HistoryDB.purchase_id,
        ProductDB.product_name,
            ProductDB.image_url,
        Purchase_HistoryDB.quantity,
        Purchase_HistoryDB.registration_date
    ).select_from(Purchase_HistoryDB).join(
        ProductDB, Purchase_HistoryDB.product_id == ProductDB.product_id
    ).filter(
        Purchase_HistoryDB.user_id == user_id
    ).order_by(
        Purchase_HistoryDB.registration_date.desc()
    ).limit(10)

    favorite_products_ids_query = db.query(
        Purchase_HistoryDB.product_id
    ).filter(
        Purchase_HistoryDB.user_id == user_id,
        Purchase_HistoryDB.favorite == True
    ).subquery()

    favorite_products_query = db.query(
        ProductDB.product_id,
        ProductDB.product_name,
        ProductDB.including_tax_price
    ).filter(
        ProductDB.product_id.in_(favorite_products_ids_query)
    )

    # クエリ実行
    recent_purchases = recent_purchases_query.all()
    favorite_products = favorite_products_query.all()

    # レスポンスデータの構築
    return MyPageResponse(
        recent_purchases=[RecentPurchase(
            purchase_id=purchase.purchase_id,
            product_name=purchase.product_name,
            quantity=purchase.quantity,
            registration_date=purchase.registration_date
        ) for purchase in recent_purchases],
        favorite_products=[FavoriteProduct(
            product_id=favorite.product_id,
            product_name=favorite.product_name,
            including_tax_price=favorite.including_tax_price
        ) for favorite in favorite_products]
    )


@app.get("/qrcode")
async def read_products_info(qrcode: int = Query(..., description="Product QR code")):
    db = SessionLocal()
    product = db.query(ProductDB).filter_by(product_qrcode=qrcode).first()
    if product:
        # Productの情報を取得
        product_info = {
            "product_id": product.product_id,
            "product_name": product.product_name,
            "manufacturer": product.manufacturer,
            "category": product.category,
            "volume": product.volume,
            "including_tax_price": product.including_tax_price,
            "quantity": product.quantity,
            "image_url": product.image_url  # 画像URLをレスポンスに追加
        }
        db.close()
        return product_info
    else:
        db.close()
        return JSONResponse(content={"product_name": "商品がマスタ未登録です"}, status_code=404)

from fastapi import Header

# 商品情報を登録するエンドポイント
@app.post('/purchase', response_model=PurchaseHistoryResponse)
async def add_purchase_history(purchase_history: PurchaseHistoryCreate, db: Session = Depends(get_db), authorization: Optional[str] = Header(None)):
    if authorization is None:
        raise HTTPException(status_code=400, detail="Authorization header is missing.")

    scheme, _, token = authorization.partition(' ')
    if not scheme or scheme.lower() != 'bearer':
        raise HTTPException(status_code=401, detail="Invalid authentication scheme.")
    if not token:
        raise HTTPException(status_code=401, detail="Token not provided.")

    user_info = get_user_by_token(db, token)
    if not user_info:
        raise HTTPException(status_code=404, detail="User not found or invalid token")

    new_history = Purchase_HistoryDB(
        user_id=user_info.user_id,
        product_id=purchase_history.product_id,
        quantity=purchase_history.quantity,
        favorite=purchase_history.favorite,
        registration_date=purchase_history.registration_date
    )

    db.add(new_history)
    db.commit()
    db.refresh(new_history)

    return PurchaseHistoryResponse(
        purchase_id=new_history.purchase_id,
        user_id=new_history.user_id,
        product_id=new_history.product_id,
        quantity=new_history.quantity,
        favorite=new_history.favorite,
        registration_date=new_history.registration_date
    )

@router.post("/survey", response_model=List[SurveyResponseSchema])
async def submit_survey(responses: List[SurveyResponseCreate], db: Session = Depends(get_db), user: UserDB = Depends(get_current_user)):
    responses_data = []
    for response in responses:
        new_response = SurveyResponse(
            user_id=user.user_id,
            question_id=response.question_id,
            answer_text=response.answer_text,
            created_at=datetime.utcnow()
        )
        db.add(new_response)
        responses_data.append(new_response)
    db.commit()
    return responses_data

app.include_router(router)

# MongoDB サーバーに接続
client = MongoClient(MONGODB_URL)
db = client["facial_treatment_gpt"]  # データベース名を指定
collection = db["embeddings_mens_facial"] #コレクション名を指定

#MongoDBのコレクションを定義
class Vector(BaseModel):
    token:str
    input_prompt: str


#OpenAIの利用およびベクトル近似関数の定義
client2 = OpenAI()

def str2vec_openai(input_str):
    return client2.embeddings.create(input = [input_str], model="text-embedding-3-small").data[0].embedding

def similar_point_cosine(input_vec1,input_vec2):
    return 1 - cosine(input_vec1, input_vec2)

#検索の関数定義
system_algorithm = "あなたは優秀で、入力された文字を正確に分類します。出力はジャンル、特長と効果、対象肌質、使用感の順で出力します。出力する言語は日本語です。"
ata2vec_algorithm_version = "1.0.0"

def insert2mongo_str2vec_openai(input_str):
    data2vec_response = client2.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": system_algorithm},
        {"role": "user", "content": input_str},
    ]
    )
    data2vec = data2vec_response.choices[0].message.content
    embedding_response = str2vec_openai(data2vec)
    useArray_search_to_db = {
        "data":input_str,
        "product_name":"unknown",
        "data2vec":data2vec,
        "data2vec_algorithm_version":ata2vec_algorithm_version,
        "model":"text-embedding-3-small",
        "embedding":embedding_response
    }
    return useArray_search_to_db


#肌の悩みに最適な化粧水をリコメンドする
@app.post('/vector')
async def vector(vector: Vector):
    db = SessionLocal()
    input_prompt = vector.input_prompt

    user = db.query(UserDB).filter(UserDB.token == vector.token).first()

    if not user:
        # トークンに対応するユーザーが存在しない場合
        raise HTTPException(status_code=404, detail="User not found")

    # CounselingDBから該当ユーザーの最新のカウンセリングデータを取得
    latest_counseling_data = db.query(CounselingDB)\
        .filter(CounselingDB.user_id == user.user_id)\
        .order_by(CounselingDB.answer_date.desc())\
        .all()

    if not latest_counseling_data:
        # カウンセリングデータが見つからない場合
        raise HTTPException(status_code=404, detail="No counseling data available")

    # 最新の日付を取得
    latest_date = latest_counseling_data[0].answer_date

    # 最新の日付でフィルタリングした結果を取得
    latest_counseling_entries = db.query(CounselingDB)\
        .filter(CounselingDB.user_id == user.user_id)\
        .filter(CounselingDB.answer_date == latest_date)\
        .all()
    db.close()

    user_info = f'あなたの大切なお客様の{user.user_name}さんは以下の特徴や状態です。\n'
    user_info += '\n'.join(f'"{entry.question}"という質問に"{entry.answer}"と回答してます。' for entry in latest_counseling_entries)

    res = insert2mongo_str2vec_openai(input_prompt)

    result = collection.find()
    df = pd.DataFrame(result)

    vec2_args = res["embedding"]
    df["similar"] = df.embedding.apply(similar_point_cosine,input_vec2=vec2_args,)
    new_df = df.sort_values("similar", ascending=False).iloc[:3].copy()
    recommend_array = new_df["data"].to_list()

    system_recommend_str = ""
    for i in range(len(recommend_array)):
        system_recommend_str += "    " + str(i + 1) + ". " + recommend_array[i] + "\n"

    system = f"""
    あなたは美容部員・BA（ビューティアドバイザー）です。出力としてお客様の名前を文中に必ず１回入れて、
    フレンドリーな口調で、的確に、さりげなく商品を提案してください。
    商品のメーカー名を間違えないで下さい。

    #お客様の情報：
    {user_info}

    # 前提条件:
    - 以下の情報はメンズ美容商品であり、これらをおすすめするように返答する。
    - 以下美容部員・BA（ビューティアドバイザー）がおすすめしたい商品：
    {system_recommend_str}
    """

    response = client2.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": input_prompt},
        ],
        temperature=0.0,
        )
    data2vec = response.choices[0].message.content


    return {"recommend_comment": data2vec}

class CounselingDB(Base):
    __tablename__ = 'counseling'
    counseling_id = Column(Integer, primary_key=True, nullable=False)
    user_id = Column(Integer, nullable=False)
    question_id = Column(Integer, nullable=False)
    question = Column(String(255), nullable=False)
    answer = Column(String(255), nullable=False)
    answer_date =  Column(DateTime, default=datetime)

class Counseling(BaseModel):
    token: str
    question_id1: str
    question1: str
    answer1: str
    question_id2: str
    question2: str
    answer2: str
    question_id3: str
    question3: str
    answer3: str
    answer_date: datetime


@app.post('/counseling')
async def read_counseling_data(counseling: Counseling):
    db = SessionLocal()

    user = db.query(UserDB).filter(UserDB.token == counseling.token).first()

    # UTCで取得した日時をJSTに変換
    answer_date_utc = counseling.answer_date
    jst = pytz.timezone('Asia/Tokyo')
    answer_date_jst = answer_date_utc.astimezone(jst)

    # データベースオブジェクトの作成
    counseling_data = [
        CounselingDB(
            user_id=user.user_id,
            question_id=counseling.question_id1,
            question=counseling.question1,
            answer=counseling.answer1,
            answer_date=answer_date_jst
        ),
        CounselingDB(
            user_id=user.user_id,
            question_id=counseling.question_id2,
            question=counseling.question2,
            answer=counseling.answer2,
            answer_date=answer_date_jst
        ),
        CounselingDB(
            user_id=user.user_id,
            question_id=counseling.question_id3,
            question=counseling.question3,
            answer=counseling.answer3,
            answer_date=answer_date_jst
        )
    ]

    # トレードをデータベースに追加してコミット
    db.add_all(counseling_data)
    db.commit()

    return {'message': 'Counseling Data added successfully'}


# データベースのセットアップ
Base.metadata.create_all(bind=engine)
