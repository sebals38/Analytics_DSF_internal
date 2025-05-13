import streamlit as st
import pandas as pd
import numpy as np
import io, os
import sqlite3
from pages.func.base_func import  *

# DataFrame 출력 형식 설정: float 형식의 소수점 자리를 1로 제한
pd.options.display.float_format = '{:.1f}'.format

st.title("I&R - Nielsen Discover Data Analysis")

def save_df_to_db(df, db_name="uploaded_data.db", table_name="analyzed_data", db_folder="db", fill_value=0):
    """
    DataFrame을 SQLite 데이터베이스의 지정된 폴더에 저장합니다.
    """
    try:
        # 데이터베이스 폴더가 없으면 생성
        os.makedirs(db_folder, exist_ok=True)
        db_path = os.path.join(db_folder, db_name)
        conn = sqlite3.connect(db_path)
        df = df.replace({None:np.nan})
        df = df.fillna(fill_value)  # NaN 값을 fill_value로 채우기
        df.to_sql(table_name, conn, if_exists='replace', index=False)
        conn.commit()
        conn.close()
        st.success(f"DataFrame을 '{db_path}' 데이터베이스의 '{table_name}' 테이블에 저장했습니다.")
        return True
    except Exception as e:
        st.error(f"데이터베이스 저장 중 오류 발생: {e}")
        return False

def load_df_from_db(db_name="uploaded_data.db", table_name="analyzed_data", db_folder="db"):
    """
    SQLite 데이터베이스의 지정된 폴더에서 DataFrame을 불러옵니다.
    """
    try:
        db_path = os.path.join(db_folder, db_name)
        conn = sqlite3.connect(db_path)
        query = f"SELECT * FROM {table_name}"
        loaded_df = pd.read_sql(query, conn)
        conn.close()
        st.success(f"'{db_path}' 데이터베이스의 '{table_name}' 테이블에서 데이터를 불러왔습니다.")
        return loaded_df
    except Exception as e:
        st.error(f"데이터베이스 로드 중 오류 발생: {e}")
        return None

if "uploaded_df" not in st.session_state:
    st.session_state["uploaded_df"] = None
# if "df_analyzed" not in st.session_state:
#     st.session_state["df_analyzed"] = None
if "final_data_rows" not in st.session_state:
    st.session_state["final_data_rows"] = None
if "final_data_rows_loaded" not in st.session_state:
    st.session_state["final_data_rows_loaded"] = None
if "scope" not in st.session_state:
    st.session_state["scope"] = "market" # 기본값 설정
if "data_saved" not in st.session_state:
    st.session_state["data_saved"] = False

with st.sidebar:
    uploaded_file = st.file_uploader("Upload CSV or Excel file to analyze", type=["csv", "xlsx"])
    db_folder_name = "db"  # 저장할 폴더 이름
    if st.button("저장된 데이터 불러오기"): # 저장된 경우에만 활성화
        st.session_state["uploaded_df"] = load_df_from_db(db_name="I&R.db", db_folder=db_folder_name)
        df = st.session_state["uploaded_df"].copy()
        df = df.replace({np.nan: 0}).copy() # NaN을 0으로 채움

    elif uploaded_file is not None:
        try:
            file_extension = uploaded_file.name.split('.')[-1].lower()
            if file_extension == "csv":
                st.session_state["uploaded_df"] = pd.read_csv(uploaded_file, skiprows=8, index_col=False, encoding="utf-8")
                st.success("DATA file loading complete!")
            elif file_extension == "xlsx":
                st.session_state["uploaded_df"] = pd.read_excel(uploaded_file, skiprows=8, index_col=False)
                st.success("DATA file loading complete!")
            else:
                st.info("Upload CSV or Excel file to analyze.")
            # st.session_state["uploaded_file"] = uploaded_file # 파일 객체 저장

            # db_folder_name = "db"  # 저장할 폴더 이름
            fill_na_value = 0
            # 데이터베이스 저장
            if st.button("분석 결과 저장"):
                if save_df_to_db(st.session_state["uploaded_df"].copy(), db_name="I&R.db", db_folder=db_folder_name, fill_value=fill_na_value):
                    st.info(f"데이터가 '{db_folder_name}' 폴더에 저장되었습니다. 아래 버튼을 눌러 불러올 수 있습니다.")

            if st.session_state["uploaded_df"] is not None:
                df = st.session_state["uploaded_df"].copy()
                df = df.replace({np.nan: 0}).copy() # NaN을 0으로 채움
        
        except pd.errors.ParserError as e:
            st.error(f"CSV file parsing error: {e}")
        except Exception as e:
            st.error(f"Error occurred: {e}")


# if "uploaded_df" in st.session_state:
if st.session_state["uploaded_df"] is not None:
    # 원 데이터프레임 컬럼 분할
    fact_columns = [col for col in df.columns if 'Unnamed' in col]
    value_columns = [col for col in df.columns if 'Value' in col]
    volume_columns = [col for col in df.columns if 'KG' in col]


    fact_data_rows, val_data_rows, vol_data_rows = generate_columns(df, fact_columns, value_columns, volume_columns)

    col_fact = df[fact_columns].iloc[0].tolist()
    if fact_data_rows is not None and len(fact_data_rows.columns) == len(col_fact):
        fact_data_rows.columns = col_fact

    intermediate_df_val = intermediate_df(df, value_columns)
    intermediate_df_vol = intermediate_df(df, volume_columns)

    latest_mo = intermediate_df_val.columns[-1][-6:] if not intermediate_df_val.empty else ""
    month_ago = intermediate_df_val.columns[-2][-6:] if len(intermediate_df_val.columns) >= 2 else ""
    latest_mo_yag = intermediate_df_val.columns[-13][-6:] if len(intermediate_df_val.columns) >= 13 else ""

    latest_3mo = intermediate_df_val.columns[-1][-6:] if not intermediate_df_val.empty else ""
    previous_3mo = intermediate_df_val.columns[-4][-6:] if len(intermediate_df_val.columns) >= 4 else ""

    # 예시: 'Feb 25'를 입력받아 volume 월별 인덱스 찾기
    target_month = st.text_input(f"분석 기준 월을 입력하세요 (예: {latest_mo})", f"{latest_mo}")
    st.write(f"target_month 초기값: {target_month}")

    target_indices_vol = find_month_index(intermediate_df_vol, target_month)
    anal_df_vol = anal_df(intermediate_df_vol, target_indices_vol)

    target_indices_val = find_month_index(intermediate_df_val, target_month)
    anal_df_val = anal_df(intermediate_df_val, target_indices_val)

    if not anal_df_vol.empty and not anal_df_val.empty:
        # 컬럼 개수 확인 후 join
        min_cols = min(len(anal_df_vol.columns), 7)
        vol_subset = anal_df_vol.iloc[:, :min_cols].copy()
        val_subset = anal_df_val.iloc[:, :min_cols].copy()
        vol_val_df = vol_subset.join(val_subset, how='left', lsuffix='_vol', rsuffix='_val')

        price_columns_base = vol_val_df.columns[:min_cols]
        price_columns = [col.replace('volume', 'price').replace('_vol', '') for col in price_columns_base]

        anal_df_unitprice = unitprice_df(vol_val_df.copy(), price_columns)

        # final_data_rows 생성 시 데이터프레임들이 비어있지 않은지 확인
        if not fact_data_rows.empty and not anal_df_val.empty and not anal_df_vol.empty and not anal_df_unitprice.empty:
            # 조인 시 인덱스 맞춰주기 (fact_data_rows에 인덱스가 없다면 무시)
            if fact_data_rows.index.inferred_type != 'integer':
                final_data_rows = fact_data_rows.join(anal_df_val, how='right').join(anal_df_vol, how='right').join(anal_df_unitprice, how='right')
                st.session_state["final_data_rows"] = final_data_rows
                # st.write("final_data_rows 컬럼 (초기 생성):", final_data_rows.columns.tolist())
            else:
                # 인덱스를 맞춰서 join
                final_data_rows = fact_data_rows.reset_index(drop=True).join(anal_df_val.reset_index(drop=True), how='right').join(anal_df_vol.reset_index(drop=True), how='right').join(anal_df_unitprice.reset_index(drop=True), how='right')
                st.session_state["final_data_rows"] = final_data_rows
                # st.write("final_data_rows 컬럼 (초기 생성, 인덱스 reset):", final_data_rows.columns.tolist())
            
            st.session_state["final_data_rows"] = generate_df_to_analyze(st.session_state["final_data_rows"])

            # if st.session_state["df_analyzed"] is None:
            #     st.session_ state["df_analyzed"] = generate_df_to_analyze(st.session_state["final_data_rows"].copy()) # 복사본 사용

            # st.subheader("분석 결과 (일부):")
            # st.dataframe(st.session_state["df_analyzed"].head())

            # # db_folder_name = "db"  # 저장할 폴더 이름
            # fill_na_value = 0

            # # 데이터베이스 저장
            # if st.button("분석 결과 저장"):
            #     if save_df_to_db(st.session_state["final_data_rows"].copy(), db_folder=db_folder_name, fill_value=fill_na_value):
            #         st.info(f"데이터가 '{db_folder_name}' 폴더에 저장되었습니다. 아래 버튼을 눌러 불러올 수 있습니다.")
    
if st.session_state["final_data_rows"] is not None:

    if st.session_state["final_data_rows"] is not None:
        st.session_state["scope"] = st.radio("분석할 단계를 선택하세요", ["market", "segment", "manufacturer", "brand", "subbrand"], index=["market", "segment", "manufacturer", "brand", "subbrand"].index(st.session_state["scope"]))

        ## total market : 예) 3번째 컬럼부터 7번째 컬럼까지 (인덱스 2부터 6까지) 선택
        cols_to_check = st.session_state["final_data_rows"].columns[2:7]
        st.session_state["final_data_rows"] = st.session_state["final_data_rows"].replace({"0": np.nan, 0: np.nan, '': np.nan, ' ': np.nan}).copy() # NaN을 0으로 채움

        nan_condition = st.session_state["final_data_rows"][st.session_state["final_data_rows"].columns[2:7]].isnull().all(axis=1)
        market_df = st.session_state["final_data_rows"][nan_condition].copy()

        if st.session_state["scope"] == "market":
            analyzed_monthly_market_df = monthly_performances(market_df)
            analyzed_monthly_market_df.columns = ["market"]
            analyzed_3monthly_market_df = three_monthly_performances(market_df)
            analyzed_3monthly_market_df.columns = ["market"]
        elif st.session_state["scope"] == "segment":
            segment_desc = "SEGMENTC"
            cols_to_check = st.session_state["final_data_rows"].columns[3:7]
            # 선택된 컬럼들이 모두 NaN인 조건으로 행 발췌
            nan_condition = st.session_state["final_data_rows"][cols_to_check].isnull().all(axis=1)

            segment_list = st.session_state["final_data_rows"][segment_desc].dropna().unique().tolist()                    
            selected_segment = st.selectbox("분석할 세그먼트를 선택하세요.", sorted(segment_list))

            segment_df = st.session_state["final_data_rows"][(nan_condition&~st.session_state["final_data_rows"].iloc[:,2].isnull()) & (st.session_state["final_data_rows"][segment_desc]==selected_segment)]
            
            st.write(f"선택된 브랜드: {selected_segment}")

            analyzed_monthly_market_df = monthly_performances(market_df)
            analyzed_monthly_market_df.columns = ["market"]
            analyzed_3monthly_market_df = three_monthly_performances(market_df)
            analyzed_3monthly_market_df.columns = ["market"]
            analyzed_monthly_segment_df = monthly_performances(segment_df)
            analyzed_monthly_segment_df.columns = ["segment"]
            analyzed_3monthly_segment_df = three_monthly_performances(segment_df)
            analyzed_3monthly_segment_df.columns = ["segment"]
                                
            analyzed_monthly_market_segment_df = pd.merge(
                analyzed_monthly_market_df, analyzed_monthly_segment_df, on=analyzed_monthly_market_df.index)
            analyzed_monthly_market_segment_df = analyzed_monthly_market_segment_df.rename(columns={'key_0':'features'})
            
            analyzed_3monthly_market_segment_df = pd.merge(
                analyzed_3monthly_market_df, analyzed_3monthly_segment_df, on=analyzed_3monthly_market_df.index)
            analyzed_3monthly_market_segment_df = analyzed_3monthly_market_segment_df.rename(columns={'key_0':'features'})
        elif st.session_state["scope"] == "manufacturer":
            manufacturer_desc = "MANUFACTURER"
            cols_to_check = st.session_state["final_data_rows"].columns[4:7]
            # 선택된 컬럼들이 모두 NaN인 조건으로 행 발췌
            nan_condition = st.session_state["final_data_rows"][cols_to_check].isnull().all(axis=1)

            all_manufacturer_df = st.session_state["final_data_rows"][nan_condition & ~((st.session_state["final_data_rows"].iloc[:,2].isnull()) | (st.session_state["final_data_rows"].iloc[:,3].isnull()))]
            
            manufacturer_list = st.session_state["final_data_rows"][manufacturer_desc].dropna().unique().tolist()                    
            selected_manufacturer = st.selectbox("분석할 제조사를 선택하세요.", sorted(manufacturer_list))

            manufacturer_df = st.session_state["final_data_rows"][(nan_condition&~st.session_state["final_data_rows"].iloc[:,2].isnull()) & (st.session_state["final_data_rows"]["MANUFACTURER"]==selected_manufacturer)]

            monthly_result_df = causal_analysis(all_manufacturer_df, timestamp=1)
            three_monthly_result_df = causal_analysis(all_manufacturer_df, timestamp=2)

            monthly_cause_diagnosis = monthly_diagnose_manufacturers(monthly_result_df, selected_manufacturer, latest_mo, month_ago)
            three_monthly_cause_diagnosis = three_monthly_diagnose_manufacturers(three_monthly_result_df, selected_manufacturer, latest_3mo, previous_3mo)

            analyzed_monthly_market_df = monthly_performances(market_df)
            analyzed_monthly_market_df.columns = ["market"]
            analyzed_3monthly_market_df = three_monthly_performances(market_df)
            analyzed_3monthly_market_df.columns = ["market"]
            analyzed_monthly_manufacturer_df = monthly_performances(manufacturer_df)
            analyzed_monthly_manufacturer_df.columns = ["manufacturer"]
            analyzed_3monthly_manufacturer_df = three_monthly_performances(manufacturer_df)
            analyzed_3monthly_manufacturer_df.columns = ["manufacturer"]
                                
            analyzed_monthly_market_manufacturer_df = pd.merge(
                analyzed_monthly_market_df, analyzed_monthly_manufacturer_df, on=analyzed_monthly_market_df.index)
            analyzed_monthly_market_manufacturer_df = analyzed_monthly_market_manufacturer_df.rename(columns={'key_0':'features'})
            
            analyzed_3monthly_market_manufacturer_df = pd.merge(
                analyzed_3monthly_market_df, analyzed_3monthly_manufacturer_df, on=analyzed_3monthly_market_df.index)
            analyzed_3monthly_market_manufacturer_df = analyzed_3monthly_market_manufacturer_df.rename(columns={'key_0':'features'})
        elif st.session_state["scope"] == "brand":
            cols_to_check = st.session_state["final_data_rows"].columns[5:7]
            # 선택된 컬럼들이 모두 NaN인 조건으로 행 발췌
            nan_condition = st.session_state["final_data_rows"][cols_to_check].isnull().all(axis=1)

            all_manufacturer_brand_df = st.session_state["final_data_rows"][nan_condition & ~((st.session_state["final_data_rows"].iloc[:,3].isnull()) | (st.session_state["final_data_rows"].iloc[:,4].isnull()))]

            selected_manufacturer_brand_list = all_manufacturer_brand_df["MANUFACTURER"].dropna().unique().tolist()                    
            selected_manufacturer = st.selectbox("분석할 제조사를 선택하세요.", sorted(selected_manufacturer_brand_list))

            selected_manufacturer_brand_df = all_manufacturer_brand_df[(nan_condition&~(all_manufacturer_brand_df.iloc[:,2].isnull() | all_manufacturer_brand_df.iloc[:,3].isnull())) & (all_manufacturer_brand_df["MANUFACTURER"]==selected_manufacturer)]

            brand_list = selected_manufacturer_brand_df["BRAND"].dropna().unique().tolist()
            
            selected_brand = st.selectbox("분석할 브랜드를 선택하세요.", sorted(brand_list))
            st.write(f"선택된 브랜드: {selected_brand}")

            brand_rows_temp = st.session_state["final_data_rows"][st.session_state["final_data_rows"]["BRAND"].str.contains(selected_brand, na=False)]
            brand_df = brand_rows_temp[brand_rows_temp["SUBBRAND"].isna()]

            segment_list = brand_df["SEGMENTC"].dropna().unique().tolist()                    
            selected_segment = st.selectbox("분석할 세그먼트를 선택하세요.", sorted(segment_list))

            brand_df = brand_df[brand_df["SEGMENTC"]==selected_segment]

            analyzed_monthly_market_df = monthly_performances(market_df)
            analyzed_monthly_market_df.columns = ["market"]
            analyzed_3monthly_market_df = three_monthly_performances(market_df)
            analyzed_3monthly_market_df.columns = ["market"]
            analyzed_monthly_brand_df = monthly_performances(brand_df)
            analyzed_monthly_brand_df.columns = ["brand"]
            analyzed_3monthly_brand_df = three_monthly_performances(brand_df)
            analyzed_3monthly_brand_df.columns = ["brand"]
                                
            analyzed_monthly_market_brand_df = pd.merge(
                analyzed_monthly_market_df, analyzed_monthly_brand_df, on=analyzed_monthly_market_df.index)
            analyzed_monthly_market_brand_df = analyzed_monthly_market_brand_df.rename(columns={'key_0':'features'})
            
            analyzed_3monthly_market_brand_df = pd.merge(
                analyzed_3monthly_market_df, analyzed_3monthly_brand_df, on=analyzed_3monthly_market_df.index)
            analyzed_3monthly_market_brand_df = analyzed_3monthly_market_brand_df.rename(columns={'key_0':'features'})
        elif st.session_state["scope"] == "subbrand":
            cols_to_check = st.session_state["final_data_rows"].columns[5:7]
            # 선택된 컬럼들이 모두 NaN인 조건으로 행 발췌
            nan_condition = st.session_state["final_data_rows"][cols_to_check].isnull().all(axis=1)

            all_manufacturer_brand_df = st.session_state["final_data_rows"][nan_condition & ~((st.session_state["final_data_rows"].iloc[:,3].isnull()) | (st.session_state["final_data_rows"].iloc[:,4].isnull()))]

            selected_manufacturer_brand_list = all_manufacturer_brand_df["MANUFACTURER"].dropna().unique().tolist()                    
            selected_manufacturer = st.selectbox("분석할 제조사를 선택하세요.", sorted(selected_manufacturer_brand_list))

            selected_manufacturer_brand_df = all_manufacturer_brand_df[(nan_condition&~(all_manufacturer_brand_df.iloc[:,2].isnull() | all_manufacturer_brand_df.iloc[:,3].isnull())) & (all_manufacturer_brand_df["MANUFACTURER"]==selected_manufacturer)]

            brand_list = selected_manufacturer_brand_df["BRAND"].dropna().unique().tolist()
            
            selected_brand = st.selectbox("분석할 브랜드를 선택하세요.", sorted(brand_list))
            st.write(f"선택된 브랜드: {selected_brand}")

            brand_rows_temp = st.session_state["final_data_rows"][st.session_state["final_data_rows"]["BRAND"].str.contains(selected_brand, na=False)]
            
            subbrand_list = brand_rows_temp[brand_rows_temp["ITEM"].isna()]["SUBBRAND"].dropna().unique().tolist()
            selected_subbrand = st.selectbox("분석할 서브브랜드를 선택하세요.", sorted(subbrand_list))
            st.write(f"선택된 서브브브랜드: {selected_subbrand}")
            
            subbrand_rows_temp = brand_rows_temp[brand_rows_temp["SUBBRAND"].str.contains(selected_subbrand, na=False)]
            subbrand_df = subbrand_rows_temp[subbrand_rows_temp["ITEM"].isna()]

            analyzed_monthly_market_df = monthly_performances(market_df)
            analyzed_monthly_market_df.columns = ["market"]
            analyzed_3monthly_market_df = three_monthly_performances(market_df)
            analyzed_3monthly_market_df.columns = ["market"]
            analyzed_monthly_subbrand_df = monthly_performances(subbrand_df)
            analyzed_monthly_subbrand_df.columns = ["subbrand"]
            analyzed_3monthly_subbrand_df = three_monthly_performances(subbrand_df)
            analyzed_3monthly_subbrand_df.columns = ["subbrand"]
                                
            analyzed_monthly_market_subbrand_df = pd.merge(
                analyzed_monthly_market_df, analyzed_monthly_subbrand_df, on=analyzed_monthly_market_df.index)
            analyzed_monthly_market_subbrand_df = analyzed_monthly_market_subbrand_df.rename(columns={'key_0':'features'})
            
            analyzed_3monthly_market_subbrand_df = pd.merge(
                analyzed_3monthly_market_df, analyzed_3monthly_subbrand_df, on=analyzed_3monthly_market_df.index)
            analyzed_3monthly_market_subbrand_df = analyzed_3monthly_market_subbrand_df.rename(columns={'key_0':'features'})

        st.write("")
        st.write("===========================================================================")        
        st.write("")

        st.subheader("Analysis Result")
        if st.session_state["scope"] == "market":
            st.write("raw data")
            st.dataframe(market_df)
            st.write("key monthly performances")
            st.dataframe(analyzed_monthly_market_df)
            st.write("key 3-monthly performances")
            st.dataframe(analyzed_3monthly_market_df)                 
        elif st.session_state["scope"] == "segment":
            st.write("raw data")
            st.dataframe(segment_df)                    
            st.write("key monthly performances")
            st.dataframe(analyzed_monthly_market_segment_df)
            st.write("key 3-monthly performances")
            st.dataframe(analyzed_3monthly_market_segment_df) 
        elif st.session_state["scope"] == "manufacturer":
            st.write("raw data")
            st.dataframe(manufacturer_df)
            st.write("key monthly performances")
            st.dataframe(analyzed_monthly_market_manufacturer_df)
            st.write("key 3-monthly performances")
            st.dataframe(analyzed_3monthly_market_manufacturer_df)
            st.write("monthly_causal_diagnosis")
            st.dataframe(monthly_cause_diagnosis)
            st.write("3monthly_causal_diagnosis")
            st.dataframe(three_monthly_cause_diagnosis)
        elif st.session_state["scope"] == "brand":
            st.write("raw data")
            st.dataframe(brand_df)
            st.write("key monthly performances")
            st.dataframe(analyzed_monthly_market_brand_df)
            st.write("key 3-monthly performances")
            st.dataframe(analyzed_3monthly_market_brand_df) 
        elif st.session_state["scope"] == "subbrand":
            st.write("raw data")
            st.dataframe(subbrand_df)                    
            st.write("key monthly performances")
            st.dataframe(analyzed_monthly_market_subbrand_df)
            st.write("key 3-monthly performances")
            st.dataframe(analyzed_3monthly_market_subbrand_df) 

    else:
            st.warning("최종 분석 결과를 생성할 데이터가 부족합니다.")
# elif uploaded_file is not None:
#         st.warning("물량 또는 금액 관련 분석 데이터를 생성할 수 없습니다. 컬럼명을 확인해주세요.")
            
# 데이터베이스에서 불러오기
# if st.button("저장된 데이터 불러오기", disabled=not st.session_state["data_saved"]): # 저장된 경우에만 활성화
# if st.button("저장된 데이터 불러오기"): # 저장된 경우에만 활성화
#     load_df_from_db(db_folder=db_folder_name)

if st.session_state["final_data_rows_loaded"] is not None:
    st.subheader("불러온 데이터:")
    st.dataframe(st.session_state["final_data_rows_loaded"])
    st.session_state["final_data_rows"] = st.session_state["final_data_rows_loaded"].copy() # 불러온 데이터를 분석에 사용
    st.session_state["final_data_rows_loaded"] = None # 불러온 데이터는 한 번만 처리

elif st.session_state["uploaded_df"] is None:
    st.info("Upload a CSV or Excel file to start analysis.")