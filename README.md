[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/D1pZhJxu)
# ML Advanced 부동산 실거래가 예측 모델링

## Team

| ![전백찬](https://avatars.githubusercontent.com/u/156163982?v=4) | ![이패캠](https://avatars.githubusercontent.com/u/156163982?v=4) | ![최패캠](https://avatars.githubusercontent.com/u/156163982?v=4) | ![김패캠](https://avatars.githubusercontent.com/u/156163982?v=4) | ![오패캠](https://avatars.githubusercontent.com/u/156163982?v=4) |
| :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: |
|            [전백찬](https://github.com/UpstageAILab)             |            [현재호](https://github.com/UpstageAILab)             |            [이진영](https://github.com/UpstageAILab)             |            [위효연](https://github.com/UpstageAILab)             |            [김주형](https://github.com/UpstageAILab)             |
|                            팀장, 데이터 수집 및 모델 학습                             |                            모델 학습 및 튜닝                             |                            모델 학습 및 튜닝                             |                            데이터 수집                             |                            데이터 수집                             |

## 1. Competiton Info

### Overview

- *House Price Prediction*_서울의 아파트 실거래가 데이터를 기반으로 아파트 실거래가를 예측하는 대회_
- 제공된 데이터뿐만 아니라 외부 데이터를 활용하는 경험
- 다양한 방법론을 실험해보면서 모델의 성능을 높이는 방법과 데이터를 분석을 경험 

### Timeline

- ex) July 9, 2024 - Start Date
- ex) July 19, 2024 - Final submission deadline

### Evaluation

- RMSE (Root Mean Squared Error)
- Public / Private 는 같은 기간 내에서 무작위로 50% 선정
![image](https://github.com/user-attachments/assets/96f0d042-5491-4b63-8f2b-b8692469e875)

## 2. Components

### Directory

새 볼륨 볼륨에 대한 폴더 경로의 목록입니다.
볼륨 일련 번호는 12CC-B2BE입니다.
F:.
│  7조.ipynb
│  apartment_gis.csv
│  apartment_info.csv
│  bus_feature.csv
│  catboost_optuna_predictions.csv
│  catboost_submission.csv
│  checkpoint
│  data.tar
│  data_with_embeddings.csv
│  data_with_tsne.csv
│  embedding.ckpt-1.data-00000-of-00001
│  embedding.ckpt-1.index
│  ET_logP.csv
│  groupby_도로명주소_건축물대장.csv
│  logs.log
│  RF_logP.csv
│  sample_submission.csv
│  subway_feature.csv
│  test.csv
│  train.csv
│  [ML Advanced] 7조.ipynb
│  [ML Advanced] 베이스라인코드 해설.ipynb
│  건축물대장.csv
│  결측치.csv
│  계약년월.csv
│  도로명_건축물.csv
│  
├─catboost_info
│  │  catboost_training.json
│  │  learn_error.tsv
│  │  time_left.tsv
│  │  
│  ├─learn
│  │      events.out.tfevents
│  │      
│  └─tmp

├─건축물대장
│      강남구.csv
│      강동구.csv
│      강북구.csv
│      강서구.csv
│      관악구.csv
│      광진구.csv
│      구로구.csv
│      금천구.csv
│      노원구.csv
│      도봉구.csv
│      동대문구.csv
│      동작구.csv
│      마포구.csv
│      서대문구.csv
│      서초구.csv
│      성동구.csv
│      성북구.csv
│      송파구.csv
│      양천구.csv
│      영등포구.csv
│      용산구.csv
│      은평구.csv
│      종로구.csv
│      중구.csv
│      중랑구.csv
│      
├─보건위생시설(현황)
│      Z_UPIS_C_UQ157.dbf
│      Z_UPIS_C_UQ157.prj
│      Z_UPIS_C_UQ157.shp
│      Z_UPIS_C_UQ157.shx
│      보건위생시설(현황).xlsx
│      
└─인허가
        서울시 공공와이파이 서비스 위치 정보.csv
        서울시 공원 및 사유지수목 위치정보 (좌표계_ WGS1984).csv
        서울시 공중화장실 위치정보 (좌표계_ WGS1984).csv
        서울시 녹지대 위치정보 (좌표계_ WGS1984).csv
        서울시 도시계획 생활권 중심지 정보 (좌표계_ GRS80).csv
        서울시 미용업 인허가 정보.csv
        서울시 병원 인허가 정보.csv
        서울시 병의원 위치 정보.csv
        서울시 식품판매업(기타) 인허가 정보(중부원점TM(EPSG2097)).csv
        서울시 유흥주점영업 인허가 정보.csv
        서울시 의원 인허가 정보.csv
        서울시 일반음식점 인허가 정보.csv
        서울시 일반음식점 인허가 정보2.csv
        학원교습소정보_2023년01월31일기준.csv
        


## 3. Data descrption

### Dataset overview

#### Train Data
- Target : 부동산 실거래가
- Shape : (111822,52)
- Time : 200701 ~ 202306
![image](https://github.com/user-attachments/assets/75e22f52-63e1-447b-8464-9ba55f9a4d78)

#### Test Data
- Target : 부동산 실거래가
- Shape : (9272, 51)
- Time : 202307 ~ 202309
![image](https://github.com/user-attachments/assets/377b433f-3d15-4e92-a823-4b18e0ef5589)


#### Extra Data
- 지하철역
![image](https://github.com/user-attachments/assets/0c178b6d-a66d-4a90-8019-121677891aa5)

- 버스정류장
![image](https://github.com/user-attachments/assets/ba97f0c2-15d4-4a79-85f9-92de53d5d2e9)



### EDA

- _Describe your EDA process and step-by-step conclusion_

![image](https://github.com/user-attachments/assets/58e5343a-a354-47f6-9e95-d4c223b466f8)

![image](https://github.com/user-attachments/assets/fda6e5c1-2ff0-4c26-a7f7-9f2ea6581c54)

![image](https://github.com/user-attachments/assets/6374a75f-fba4-40be-bece-b1c2bc7aedd2)

![image](https://github.com/user-attachments/assets/7701936a-253b-43be-b846-587304030ada)


| 항목                              | 결측률 (%)        |
|---------------------------------|-------------------|
| 시군구                           | 0.000000          |
| 번지                             | 0.020122          |
| 본번                             | 0.006648          |
| 부번                             | 0.006648          |
| 아파트명                          | 0.189346          |
| 전용면적(㎡)                       | 0.000000          |
| 계약년월                          | 0.000000          |
| 계약일                           | 0.000000          |
| 층                               | 0.000000          |
| 건축년도                          | 0.000000          |
| 도로명                           | 0.000000          |
| 해제사유발생일                      | 99.450844         |
| 등기신청일자                       | 98.508724         |
| 거래유형                          | 96.308552         |
| 중개사소재지                       | 96.624306         |
| k-단지분류(아파트,주상복합등등)         | 77.765949         |
| k-전화번호                        | 77.728452         |
| k-팩스번호                        | 77.949887         |
| 단지소개기존clob                   | 93.871433         |
| k-세대타입(분양형태)                 | 77.664184         |
| k-관리방식                        | 77.664184         |
| k-복도유형                        | 77.693348         |
| k-난방방식                        | 77.664184         |
| k-전체동수                        | 77.760098         |
| k-전체세대수                       | 77.664184         |
| k-건설사(시공사)                   | 77.798215         |
| k-시행사                         | 77.815678         |
| k-사용검사일-사용승인일              | 77.676062         |
| k-연면적                         | 77.664184         |
| k-주거전용면적                     | 77.668173         |
| k-관리비부과면적                    | 77.664184         |
| k-전용면적별세대현황(60㎡이하)         | 77.668173         |
| k-전용면적별세대현황(60㎡~85㎡이하)     | 77.668173         |
| k-85㎡~135㎡이하                   | 77.668173         |
| k-135㎡초과                       | 99.970836         |
| k-홈페이지                        | 89.843843         |
| k-등록일자                        | 98.962143         |
| k-수정일자                        | 77.668173         |
| 고용보험관리번호                     | 81.620592         |
| 경비비관리형태                      | 77.791478         |
| 세대전기계약방법                     | 78.485392         |
| 청소비관리형태                      | 77.808321         |
| 건축면적                          | 77.677835         |
| 주차대수                          | 77.677658         |
| 기타/의무/임대/임의=1/2/3/4         | 77.664184         |
| 단지승인일                         | 77.728806         |
| 사용허가여부                        | 77.664184         |
| 관리비 업로드                       | 77.664184         |
| 좌표X                           | 77.673669         |
| 좌표Y                           | 77.673669         |
| 단지신청일                         | 77.669680         |
| target                          | 0.821917          |
| is_test                         | 0.000000          |


결측치가 매우 많은 데이터셋의 모습 -> 외부데이터를 통해 파생변수의 생성이 필요

좌표 X와 좌표 Y colum의 결측치를 Geocoding을 통해 채워넣기
![image](https://github.com/user-attachments/assets/c0dc3bbc-f8ab-448c-baf7-ece580320970)
![브이월드](https://www.vworld.kr/dev/v4dv_geocoderguide2_s001.do)
![image](https://github.com/user-attachments/assets/7dff7404-228b-4677-8f6c-fac4dcf6e42f)
![image](https://github.com/user-attachments/assets/edd6a197-83b0-4518-bf57-8f350e6c6fb3)
![image](https://github.com/user-attachments/assets/bd21a6f4-e13e-4eba-b8b8-0c2d2564c324)


### Feature engineering

#### 파생변수 생성
1. 시군구 Feature에 대해 '구', '동' 변수를 추출하여 Label Encoding
2. 계약년월 Feature에 대해 계약년도, 계약월 feature 생성
3. 계약년도 - 건축년도 = 아파트 년수 Feature 생성
4. 도로명 주소에 대해 Ko-GPT2 기반의 문장 Encoding 을 통해 786개의 벡터열을 생성하고 T-SNE로 축소하여 3개의 Enbedding Vector를 생성


#### 외부데이터 활용

![서울집합건물통합정보마당](https://openab.seoul.go.kr/build/info.do?gubun=document)
![image](https://github.com/user-attachments/assets/403d80b5-4fcc-47bb-82dc-48c121b5e45c)


![서울 열린데이터 광장](https://data.seoul.go.kr/dataList/OA-16094/S/1/datasetView.do)
![image](https://github.com/user-attachments/assets/24d0de91-33d5-4c6b-89f9-6002ec8996eb)


![한국은행경제통계시스템](https://ecos.bok.or.kr/#/SearchStat)
![image](https://github.com/user-attachments/assets/4aa9f74d-0e7b-49fc-a96e-c729423c373b)

을 통해 외부데이터를 끌어와서 도로명주소와 계약년월 데이터를 기반으로 결합

X, Y 좌표를 활용해서 아파트 주변 버스, 지하철, 음싞점, 병원, 공원에 대해 500m, 1000m 내에 위치한 숫자를 변수로 생성

![image](https://github.com/user-attachments/assets/cf7f2cda-9ada-4ed4-b749-e6e93dcbae8c)

결합한 데이터셋

| 번호 | 열 이름                     |
|----|-----------------------------|
| 1  | 아파트명                    |
| 2  | 전용면적                    |
| 3  | 계약일                     |
| 4  | 층                        |
| 5  | 건축년도                    |
| 6  | target                   |
| 7  | 좌표X_y                   |
| 8  | 좌표Y_y                   |
| 9  | bus_stop_count_500m       |
| 10 | subway_count_500m         |
| 11 | bus_stop_count_1000m      |
| 12 | subway_count_1000m        |
| 13 | restaurant_count_500m     |
| 14 | hospital_count_500m       |
| 15 | park_count_500m           |
| 16 | park_count_1000m          |
| 17 | 아파트매매가격지수             |
| 18 | 뉴스심리지수                  |
| 19 | 경제심리지수                  |
| 20 | GDP                      |
| 21 | GNI                      |
| 22 | 코스피시가총액                |
| 23 | 코스피거래량                  |
| 24 | 코스피종가                   |
| 25 | 주택담보대출                  |
| 26 | 국민주택채권금리               |
| 27 | 현금통화                    |
| 28 | 예금기관부채                  |
| 29 | 구                        |
| 30 | 동                        |
| 31 | 계약년도                    |
| 32 | 계약월                     |
| 33 | 건축년수                    |
| 34 | 건축면적                    |
| 35 | 연면적                     |
| 36 | 용적률산정연면적               |
| 37 | 세대수                     |
| 38 | 지상층수                    |
| 39 | 지하층수                    |
| 40 | 승용승강기수                  |
| 41 | 비상용승강기수                |
| 42 | embedding_0              |
| 43 | embedding_1              |
| 44 | embedding_2              |

![image](https://github.com/user-attachments/assets/298bf923-db13-4c7a-86c8-45709bb3b9af)
![image](https://github.com/user-attachments/assets/b3ef11c0-7142-45f7-9c50-57f1ddafb58e)
![image](https://github.com/user-attachments/assets/a02ae7a8-5f5c-42b7-a7cb-a80d5d661d77)
![image](https://github.com/user-attachments/assets/b3385e39-08d1-4cf4-aaf1-3a6bcb973f76)
![image](https://github.com/user-attachments/assets/cbeb37fd-11b7-4107-a4ad-41427748a1ab)
![image](https://github.com/user-attachments/assets/74219a59-bb81-4506-b989-ebfe6056c1f3)
![image](https://github.com/user-attachments/assets/a9459027-41db-463d-8cbf-e0d42e16143e)
![image](https://github.com/user-attachments/assets/866dfc56-d31b-4664-9427-26ef981d6400)
![image](https://github.com/user-attachments/assets/a2d54116-b972-4a95-a296-66ccfa032173)

Feature들 중에서 죄편향된 데이터들에 대해 Log1p를 취해준 파생변수를 생성

![image](https://github.com/user-attachments/assets/c6ab94ce-e46b-41e9-b197-31920683444c)
![image](https://github.com/user-attachments/assets/e417787b-41dd-4f48-8b68-69ae03bf61ba)
![image](https://github.com/user-attachments/assets/ffe81628-695a-4902-ab72-17aa6adad1d7)

정규분포에 가깝게 변환하여 파생변수를 생성


#### Target 인코딩

![image](https://github.com/user-attachments/assets/1fa36abe-1069-4d1d-bc87-78f07ef6fc9a)

타겟인 실거래가격이 좌편향된 데이터임을 확인

np.log1p 를 통해서 타겟데이터를 인코딩

![image](https://github.com/user-attachments/assets/3fb181ad-35e8-41f1-86f1-a7338ec955fc)

타겟 데이터의 분포를 변형

## 4. Modeling

### Model descrition
최종학습 데이터셋
| 번호 | 열 이름                        |
|----|------------------------------|
| 1  | 아파트명                      |
| 2  | 전용면적                      |
| 3  | 계약일                       |
| 4  | 층                          |
| 5  | 건축년도                      |
| 6  | target                     |
| 7  | 좌표X_y                     |
| 8  | 좌표Y_y                     |
| 9  | bus_stop_count_500m         |
| 10 | subway_count_500m           |
| 11 | bus_stop_count_1000m        |
| 12 | subway_count_1000m          |
| 13 | restaurant_count_500m       |
| 14 | hospital_count_500m         |
| 15 | park_count_500m             |
| 16 | park_count_1000m            |
| 17 | 아파트매매가격지수               |
| 18 | 뉴스심리지수                    |
| 19 | 경제심리지수                    |
| 20 | GDP                        |
| 21 | GNI                        |
| 22 | 코스피시가총액                  |
| 23 | 코스피거래량                    |
| 24 | 코스피종가                     |
| 25 | 주택담보대출                    |
| 26 | 국민주택채권금리                 |
| 27 | 현금통화                      |
| 28 | 예금기관부채                    |
| 29 | 구                          |
| 30 | 동                          |
| 31 | 계약년도                      |
| 32 | 계약월                       |
| 33 | 건축년수                      |
| 34 | 건축면적                      |
| 35 | 연면적                       |
| 36 | 용적률산정연면적                 |
| 37 | 세대수                       |
| 38 | 지상층수                      |
| 39 | 지하층수                      |
| 40 | 승용승강기수                    |
| 41 | 비상용승강기수                  |
| 42 | embedding_0                |
| 43 | embedding_1                |
| 44 | embedding_2                |
| 45 | bus_stop_count_500m_log1p   |
| 46 | subway_count_500m_log1p     |
| 47 | bus_stop_count_1000m_log1p  |
| 48 | subway_count_1000m_log1p    |
| 49 | restaurant_count_500m_log1p |
| 50 | hospital_count_500m_log1p   |
| 51 | park_count_500m_log1p       |
| 52 | park_count_1000m_log1p      |
| 53 | GDP_log1p                   |
| 54 | GNI_log1p                   |
| 55 | 코스피시가총액_log1p            |
| 56 | 코스피거래량_log1p              |
| 57 | 코스피종가_log1p               |
| 58 | 주택담보대출_log1p              |
| 59 | 현금통화_log1p                |
| 60 | 예금기관부채_log1p              |
| 61 | 건축면적_log1p                |
| 62 | 연면적_log1p                  |
| 63 | 용적률산정연면적_log1p           |
| 64 | 세대수_log1p                  |
| 65 | 지상층수_log1p                |
| 66 | 지하층수_log1p                |
| 67 | 승용승강기수_log1p              |
| 68 | 비상용승강기수_log1p            |




### Modeling Process

- _Write model train and test process with capture_


pycaret 라이브러리를 통해 데이터셋에 대해 AutoML을 진행
![image](https://github.com/user-attachments/assets/6b73e556-70ae-4a2a-af85-55c274e57a77)

Random Forest Regressor
Extra Trees Regressor
CatBoost Regressor
가 좋은 성능을 보이는 것을 확인

optuna를 통해 각 모델에 대해 하이퍼파라미터를 튜닝

각 모델별로 모델을 생성하여 제출후 리더보드 성적을 확인
![image](https://github.com/user-attachments/assets/5ca4385e-988b-4dfa-a344-564889118a92)

모델을 앙상블 한 것보다 Extra Trees Regressor 모델 단독으로 사용했을 때 가장 좋은 결과를 나타냄



## 5. Result

![image](https://github.com/user-attachments/assets/509d2696-ec91-4a02-a88a-411bfd328044)

### Leader Board

#### Public
![image](https://github.com/user-attachments/assets/49173ed5-603f-4dab-82bf-1e4decdd95a8)

12조 중 11등

#### Private
![image](https://github.com/user-attachments/assets/b4d2fb0f-5bee-4c0f-91c6-8c2eb394ddcb)

12조 중 7등

으로 Private에서의 일반화 성능이 더 뛰어나게 나온 것으로 확인

### Presentation

- [_Insert your presentaion file(pdf) link_](https://docs.google.com/presentation/d/1dEIkEcMeLf1H8MTlXlODuT4bRB68rQtD/edit?usp=drive_link&ouid=107186399434589793386&rtpof=true&sd=true)
