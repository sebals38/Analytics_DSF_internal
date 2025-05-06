import streamlit as st
import pandas as pd
import numpy as np
import io
from pages.func.base_func import *

# DataFrame 출력 형식 설정: float 형식의 소수점 자리를 1로 제한
pd.options.display.float_format = '{:.1f}'.format

st.title("CM - Nielsen Discover Data Analysis")

with st.sidebar:
    uploaded_file = st.file_uploader("CSV 또는 Excel 파일 업로드", type=["csv", "xlsx"])

if uploaded_file is not None:
    file_extension = uploaded_file.name.split('.')[-1].lower()
    try:
        # 업로드된 파일을 pandas DataFrame으로 읽기
        if file_extension == "csv":
            uploaded_df = pd.read_csv(uploaded_file, skiprows=8, index_col=False, encoding="utf-8")
            st.success("DATA 파일 로드 완료!")
        elif file_extension == "xlsx":
            uploaded_df = pd.read_excel(uploaded_file, skiprows=8, index_col=False)
            st.success("DATA 파일 로드 완료!")

        # 데이터 분석 함수 호출
        df = generate_df_to_analyze(uploaded_df)

        # 원 데이터프레임 컬럼 분할
        fact_columns = [col for col in df.columns if 'Unnamed' in col]
        value_columns = [col for col in df.columns if 'Value' in col]
        volume_columns = [col for col in df.columns if 'KG' in col]
        
        fact_data_rows, val_data_rows, vol_data_rows = generate_columns(df, fact_columns, value_columns, volume_columns)

        col_fact = df[fact_columns].iloc[0]
        fact_data_rows.columns = col_fact

        intermediate_df_val = intermediate_df(df, value_columns)
        intermediate_df_vol = intermediate_df(df, volume_columns)

        latest_mo = intermediate_df_val.columns[-1][-6:]
        month_ago = intermediate_df_val.columns[-2][-6:]
        latest_mo_yag = intermediate_df_val.columns[-13][-6:]

        latest_3mo = intermediate_df_val.columns[-1][-6:]
        previous_3mo = intermediate_df_val.columns[-4][-6:]

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

            anal_df_unitprice = vol_val_df.copy()
            anal_df_unitprice = unitprice_df(anal_df_unitprice, price_columns)
            
            # final_data_rows 생성 시 anal_df가 비어있지 않은지 확인
            if not fact_data_rows.empty and not anal_df_val.empty and not anal_df_vol.empty and not anal_df_unitprice.empty:
                # 조인 시 인덱스 맞춰주기 (fact_data_rows에 인덱스가 없다면 무시)
                if fact_data_rows.index.inferred_type != 'integer':
                    final_data_rows = fact_data_rows.join(anal_df_val, how='right').join(anal_df_vol, how='right').join(anal_df_unitprice, how='right')
                    st.write(final_data_rows.columns)
                else:
                    # 인덱스를 맞춰서 join
                    final_data_rows = fact_data_rows.reset_index(drop=True).join(anal_df_val.reset_index(drop=True), how='right').join(anal_df_vol.reset_index(drop=True), how='right').join(anal_df_unitprice.reset_index(drop=True), how='right')
                
                scope = st.radio("분석할 단계를 선택하세요", ["market", "manufacturer", "manufacturer/segment", "manufacturer/brand", "subbrand"])

                
                ## total market : 예) 3번째 컬럼부터 7번째 컬럼까지 (인덱스 2부터 6까지) 선택
                cols_to_check = final_data_rows.columns[3:9]

                # 선택된 컬럼들이 모두 NaN인 조건으로 행 발췌
                nan_condition = final_data_rows[cols_to_check].isnull().all(axis=1)
                market_df = final_data_rows[nan_condition & (final_data_rows["VARIANT"]=="GENERAL")]

                if scope == "market":
                    analyzed_monthly_market_df = monthly_performances(market_df)
                    analyzed_monthly_market_df.columns = ["market"]
                    analyzed_3monthly_market_df = three_monthly_performances(market_df)
                    analyzed_3monthly_market_df.columns = ["market"]
                # elif scope == "segment":
                #     segment_desc = "SEGMENTB"
                #     cols_to_check = final_data_rows.columns[4:9]
                #     # 선택된 컬럼들이 모두 NaN인 조건으로 행 발췌
                #     nan_condition = final_data_rows[cols_to_check].isnull().all(axis=1)

                #     segment_list = final_data_rows[segment_desc].dropna().unique().tolist()                    
                #     selected_segment = st.selectbox("분석할 세그먼트를 선택하세요.", sorted(segment_list))

                #     segment_df = final_data_rows[(nan_condition&~final_data_rows.iloc[:,2].isnull()) & (final_data_rows[segment_desc]==selected_segment)]
                    
                #     st.write(f"선택된 브랜드: {selected_segment}")

                #     analyzed_monthly_segment_df = monthly_performances(segment_df)
                #     analyzed_3monthly_segment_df = three_monthly_performances(segment_df)
                elif scope == "manufacturer":
                    cols_to_check = final_data_rows.columns[4:9]
                    # 선택된 컬럼들이 모두 NaN인 조건으로 행 발췌
                    nan_condition = final_data_rows[cols_to_check].isnull().all(axis=1)

                    all_manufacturer_df = final_data_rows[nan_condition & ~((final_data_rows.iloc[:,2].isnull()) | (final_data_rows.iloc[:,3].isnull()))]
                    
                    manufacturer_list = final_data_rows["MANUFACTURER"].dropna().unique().tolist()                    
                    selected_manufacturer = st.selectbox("분석할 제조사를 선택하세요.", sorted(manufacturer_list))

                    manufacturer_df = final_data_rows[nan_condition & (final_data_rows["MANUFACTURER"]==selected_manufacturer)]
                    # manufacturer_df = final_data_rows[(nan_condition&~(final_data_rows.iloc[:,2].isnull() | final_data_rows.iloc[:,3].isnull())) & (final_data_rows["MANUFACTURER"]==selected_manufacturer)]

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

                elif scope == "manufacturer/segment":
                    cols_to_check = final_data_rows.columns[5:9]
                    # 선택된 컬럼들이 모두 NaN인 조건으로 행 발췌
                    nan_condition = final_data_rows[cols_to_check].isnull().all(axis=1)

                    manufacturer_list = final_data_rows["MANUFACTURER"].dropna().unique().tolist()                    
                    selected_manufacturer = st.selectbox("분석할 제조사를 선택하세요.", sorted(manufacturer_list))

                    manufacturer_df = final_data_rows[(nan_condition&~(final_data_rows.iloc[:,2].isnull() | final_data_rows.iloc[:,3].isnull())) & (final_data_rows["MANUFACTURER"]==selected_manufacturer)]

                    manufacturer_segment_list = manufacturer_df["SEGMENTB"].dropna().unique().tolist()
                    
                    if len(manufacturer_segment_list) > 1:
                        selected_manufacturer_segment = st.selectbox("제조사내 분석할 세그먼트를 선택하세요.", sorted(manufacturer_segment_list))
                        segment_df = manufacturer_df[manufacturer_df["SEGMENTB"]==selected_manufacturer_segment]
                    
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

                elif scope == "manufacturer/brand":
                    cols_to_check = final_data_rows.columns[6:9]
                    # 선택된 컬럼들이 모두 NaN인 조건으로 행 발췌
                    nan_condition = final_data_rows[cols_to_check].isnull().all(axis=1)

                    all_manufacturer_brand_df = final_data_rows[nan_condition & ~((final_data_rows.iloc[:,2].isnull()) | (final_data_rows.iloc[:,3].isnull()) | (final_data_rows.iloc[:,4].isnull()) | (final_data_rows.iloc[:,5].isnull()))]

                    selected_manufacturer_brand_list = all_manufacturer_brand_df["MANUFACTURER"].dropna().unique().tolist()                    
                    selected_manufacturer = st.selectbox("분석할 제조사를 선택하세요.", sorted(selected_manufacturer_brand_list))

                    selected_manufacturer_brand_df = all_manufacturer_brand_df[(nan_condition&~(all_manufacturer_brand_df.iloc[:,2].isnull() | all_manufacturer_brand_df.iloc[:,3].isnull())) & (all_manufacturer_brand_df["MANUFACTURER"]==selected_manufacturer)]
                    
                    brand_list = selected_manufacturer_brand_df["BRAND"].dropna().unique().tolist()
                    
                    selected_brand = st.selectbox("분석할 브랜드를 선택하세요.", sorted(brand_list))
                    st.write(f"선택된 브랜드: {selected_brand}")

                    brand_rows_temp = final_data_rows[final_data_rows["BRAND"].str.contains(selected_brand, na=False)]
                    brand_df = brand_rows_temp[brand_rows_temp["SUBBRAND"].isna()]

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
                    
                elif scope == "subbrand":
                    # brand_list = final_data_rows["BRAND"].dropna().unique().tolist()
                    
                    # selected_brand = st.selectbox("분석할 브랜드를 선택하세요.", sorted(brand_list))
                    # st.write(f"선택된 브랜드: {selected_brand}")

                    cols_to_check = final_data_rows.columns[6:9]
                    # 선택된 컬럼들이 모두 NaN인 조건으로 행 발췌
                    nan_condition = final_data_rows[cols_to_check].isnull().all(axis=1)

                    all_manufacturer_brand_df = final_data_rows[nan_condition & ~((final_data_rows.iloc[:,2].isnull()) | (final_data_rows.iloc[:,3].isnull()) | (final_data_rows.iloc[:,4].isnull()) | (final_data_rows.iloc[:,5].isnull()))]

                    selected_manufacturer_brand_list = all_manufacturer_brand_df["MANUFACTURER"].dropna().unique().tolist()                    
                    selected_manufacturer = st.selectbox("분석할 제조사를 선택하세요.", sorted(selected_manufacturer_brand_list))

                    selected_manufacturer_brand_df = all_manufacturer_brand_df[(nan_condition&~(all_manufacturer_brand_df.iloc[:,2].isnull() | all_manufacturer_brand_df.iloc[:,3].isnull())) & (all_manufacturer_brand_df["MANUFACTURER"]==selected_manufacturer)]
                    
                    brand_list = selected_manufacturer_brand_df["BRAND"].dropna().unique().tolist()
                    
                    selected_brand = st.selectbox("분석할 브랜드를 선택하세요.", sorted(brand_list))
                    st.write(f"선택된 브랜드: {selected_brand}")

                    brand_rows_temp = final_data_rows[final_data_rows["BRAND"].str.contains(selected_brand, na=False)]
                    # brand_df = brand_rows_temp[brand_rows_temp["SUBBRAND"].isna()]
                    # st.write(brand_rows_temp)
                    subbrand_list = brand_rows_temp[brand_rows_temp["ITEM"].isna()]["SUBBRAND"].dropna().unique().tolist()
                    selected_subbrand = st.selectbox("분석할 서브브랜드를 선택하세요.", sorted(subbrand_list))
                    st.write(f"선택된 서브브브랜드: {selected_subbrand}")
                    
                    subbrand_rows_temp = brand_rows_temp[brand_rows_temp["SUBBRAND"] == selected_subbrand]
                    # subbrand_rows_temp = brand_rows_temp[brand_rows_temp["SUBBRAND"].str.contains(selected_subbrand, na=False)]
                    if "SUBSEGMENT" in subbrand_rows_temp.columns:
                        subbrand_df = subbrand_rows_temp[subbrand_rows_temp["ITEM"].isna() & subbrand_rows_temp["SUBSEGMENT"].isna()]
                    else:
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
                    
                    
                    # st.write(subbrand_df)

                # st.markdown(
                #     """
                #     <style>
                #     h2 {
                #         text-align: left !important;
                #         background-color: lightblue; /* 확인용 배경색 */
                #         padding-left: 10px;
                #     }
                #     .stDataFrame {
                #         width: 100% !important;
                #         border: 2px solid red; /* 확인용 테두리 */
                #     }
                #     </style>
                #     """,
                #     unsafe_allow_html=True,
                # )

                st.write("")
                st.write("===========================================================================")        
                st.write("")                

                st.subheader("Analysis Result")
                if scope == "market":
                    st.write("raw data")
                    st.dataframe(market_df)
                    st.write("key monthly performances")
                    st.dataframe(analyzed_monthly_market_df)
                    st.write("key 3-monthly performances")
                    st.dataframe(analyzed_3monthly_market_df) 
                # elif scope == "segment":
                #     st.write("raw data")
                #     st.dataframe(segment_df)                    
                #     st.write("key monthly performances")
                #     st.dataframe(analyzed_monthly_segment_df)
                #     st.write("key 3-monthly performances")
                #     st.dataframe(analyzed_3monthly_segment_df) 
                elif scope == "manufacturer":
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
                elif scope == "manufacturer/segment":
                    st.write("raw data")
                    st.dataframe(manufacturer_df)                    
                    st.write("key monthly performances")
                    st.dataframe(analyzed_monthly_market_segment_df)
                    st.write("key 3-monthly performances")
                    st.dataframe(analyzed_3monthly_market_segment_df)
                elif scope == "manufacturer/brand":
                    st.write("raw data")
                    st.dataframe(brand_df)
                    st.write("key monthly performances")
                    st.dataframe(analyzed_monthly_market_brand_df)
                    st.write("key 3-monthly performances")
                    st.dataframe(analyzed_3monthly_market_brand_df) 
                elif scope == "subbrand":
                    st.write("raw data")
                    st.dataframe(subbrand_df)                    
                    st.write("key monthly performances")
                    st.dataframe(analyzed_monthly_market_subbrand_df)
                    st.write("key 3-monthly performances")
                    st.dataframe(analyzed_3monthly_market_subbrand_df) 
            else:
                st.warning("최종 분석 결과를 생성할 데이터가 부족합니다.")
        elif uploaded_file is not None:
            st.warning("물량 또는 금액 관련 분석 데이터를 생성할 수 없습니다. 컬럼명을 확인해주세요.")

    except pd.errors.ParserError as e:
        st.error(f"CSV 파일 파싱 오류: {e}")
    except Exception as e:
        st.error(f"오류 발생: {e}")

else:
    st.info("분석할 CSV 또는 Excel 파일을 업로드해주세요.")