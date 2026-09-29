# Toss Close Scanner v1

토스증권 Open API를 이용해 장 마감 후보를 스캔하고,
다음 거래일 상승 여부를 백테스트하는 1차 버전입니다.

## 기능
- 토스증권 OAuth2 Client Credentials 인증
- 일봉 OHLCV 조회
- 기술적 특징 계산
- 규칙 기반 종가매매 점수(0~100)
- 후보 TOP N 출력
- 과거 데이터 CSV 기반 백테스트
- 주문 API는 포함하지 않음

## 설치
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

`.env.example`을 `.env`로 복사하고 토스증권 Open API의 client_id/client_secret을 입력하세요.

## 종목 유니버스
`data/universe.csv`:
```csv
symbol,name
005930,삼성전자
000660,SK하이닉스
035420,NAVER
```

## 실행
```bash
python -m src.main scan --top 10
python -m src.main scan --symbols 005930 000660 035420 --top 10
python -m src.main backtest --data-dir data/history
```

CSV 백테스트 파일 형식:
```csv
timestamp,open,high,low,close,volume
2026-01-02,180000,183000,178000,182000,12000000
```

## 주의
현재 `score`는 규칙 기반 점수이며 `display_probability`는 실제 확률이 아닙니다.
실제 상승확률 모델은 충분한 과거 데이터로 out-of-sample 검증과 확률 보정을 거쳐야 합니다.

## 웹 스캐너 실행

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

브라우저에서 종가매매 AI 스캐너가 열립니다. 현재 v1은 한국주식 저장 데이터 기반이며, 모델 점수는 실제 확률이 아닌 순위용 추정치입니다.

## v3.2 변경사항
- 종목명 대신 코드만 표시되던 항목을 현재 KRX 종목명으로 자동 보완합니다(인터넷 연결 시).
- 사이드바에 `종목 직접 검색` 기능을 추가했습니다.
- `삼성전자` 또는 `005930`처럼 종목명/코드를 입력하면 오늘 TOP 신호 포함 여부, 모델 점수, RSI, 거래량 비율, 최근 5일 수익률, 거래대금, 최근 차트를 확인할 수 있습니다.
- 로컬 499종목에 없는 KOSPI/KOSDAQ 종목도 인터넷 연결 시 일봉을 받아 저장된 모델로 별도 분석합니다.
- 검색 결과의 모델 점수는 실제 상승확률이 아닙니다.
