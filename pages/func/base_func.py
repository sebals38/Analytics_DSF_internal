import streamlit as st
import pandas as pd
import numpy as np
import io

# DataFrame 출력 형식 설정: float 형식의 소수점 자리를 1로 제한
pd.options.display.float_format = '{:.1f}'.format

def generate_df_to_analyze(df):
    """활성영역 (마지막 row) 검출하여 활용대상 데이터프레임 생성"""
    df = df.replace({"0": 0, np.nan:0, '': 0, ' ': 0}).copy() # NaN을 0으로 채움
    space_index = None
    for index, value in enumerate(df.iloc[:, 0]):
        # if pd.isna(value):
        if (value==0):
            space_index = index
            break

    # 공백이 없는 경우 전체 데이터프레임 사용
    if space_index is None:
        new_df = df
    else:
        # 데이터프레임 슬라이싱 및 새로운 데이터프레임 생성
        new_df = df.loc[:space_index - 1, :]
    return new_df

def generate_columns(df, fact_columns, value_columns, volume_columns):
    """데이터프레임에서 팩트, 금액, 물량 컬럼 데이터 추출"""
    df[value_columns] = df[value_columns].map(lambda x: str(x).replace('₩', ''))

    fact_data_rows = df[fact_columns].iloc[1:].copy()
    val_data_rows = df[value_columns].iloc[1:].copy()
    vol_data_rows = df[volume_columns].iloc[1:].copy()

    return fact_data_rows, val_data_rows, vol_data_rows

def intermediate_df(df, columns):
    """월별 및 3개월 합산/MS 중간 데이터프레임 생성"""
    if columns[0] == 'Sales (KG)':
        frame_of_reference = 'volume'
    else:
        frame_of_reference = 'value'

    # 월년 (ex, Jan 25) 추출
    month_year_col = df[columns].iloc[0].map(lambda x: f'{frame_of_reference}_{x[:6]}')

    # 3개월 물량, 금액 컬럼 추가
    data_rows = df[columns].iloc[1:].copy()
    data_rows.columns = month_year_col

    # 1000단위 콤마 제거 및 숫자 변환
    for col in data_rows.columns:
        data_rows[col] = data_rows[col].astype(str).str.replace(',', '').astype(float)

    # 월간 MS 데이터프레임 작성
    monthly_MS = {}
    for i, col_name in enumerate(data_rows.columns): # 명시적으로 인덱스 사용하지 않음
        first_row = data_rows[col_name].iloc[0]
        monthly_MS_col = f'MS_{frame_of_reference}_{col_name[-6:]}'
        monthly_MS[monthly_MS_col] = ((data_rows[col_name].iloc[0:] / first_row) * 100).round(1)
    monthly_MS_df = pd.DataFrame(monthly_MS)

    # 3개월 합산 데이터
    three_monthly_data = {}
    for i in range(len(month_year_col) - 2):
        group_cols = month_year_col.iloc[i:i + 3]
        col_3mo_val = f'3mo_{month_year_col.iloc[i + 2]}'
        three_monthly_data[col_3mo_val] = data_rows.loc[:, group_cols].sum(axis=1)
    three_monthly_df = pd.DataFrame(three_monthly_data)

    # 3개월 MS 데이터프레임 작성
    three_monthly_MS_data = {}
    for i, col_name in enumerate(three_monthly_df.columns): # 명시적으로 인덱스 사용하지 않음
        first_row = three_monthly_df[col_name].iloc[0]
        three_monthly_MS_col = f'3mo_MS_{frame_of_reference}_{col_name[-6:]}'
        three_monthly_MS_data[three_monthly_MS_col] = ((three_monthly_df[col_name].iloc[0:] / first_row) * 100).round(1)
    three_monthly_MS_df = pd.DataFrame(three_monthly_MS_data)

    intermediate_df = data_rows.join(three_monthly_df, how="left").join(monthly_MS_df, how="left").join(three_monthly_MS_df, how="left")
    return intermediate_df

def anal_df(df, target_indices):
    """특정 컬럼 인덱스를 기준으로 데이터프레임 추출"""
    original_column_list = list(df.columns)
    if target_indices is not None:
        target_list = [original_column_list[i] for i in target_indices]
        return df[target_list]
    else:
        return pd.DataFrame()

def unitprice_df(anal_df_unitprice, price_columns):
    """물량과 금액 데이터를 이용하여 단가 데이터프레임 생성"""
    for i in range(len(price_columns)):
        if i + 7 < len(anal_df_unitprice.columns) and i < len(anal_df_unitprice.columns):
            # ZeroDivisionError 처리 추가
            column_name = price_columns[i]
            numerator = anal_df_unitprice.iloc[:, i + 7]
            denominator = anal_df_unitprice.iloc[:, i]
            price = numerator.div(denominator, fill_value=0) # 분모가 0일 경우 NaN으로 처리
            # anal_df_unitprice[column_name] = f"{price:,.1f}"
            anal_df_unitprice[column_name] = price.apply(lambda x: f"{x:,.1f}" if pd.notna(x) else "")
    return anal_df_unitprice.drop(anal_df_unitprice.columns[0:14], axis=1, errors='ignore')

def find_month_index(df, target_month):
    """
    특정 시점을 입력받아 관련 컬럼들의 인덱스를 반환합니다.

    Args:
        df (pd.DataFrame): 분석할 데이터프레임입니다.
        target_month (str): 찾고자 하는 기준 월 (예: 'Feb 25').

    Returns:
        tuple: 관련 컬럼들의 인덱스 튜플 (month_index, ...). 찾지 못하면 None을 포함한 튜플을 반환합니다.
    """
    if not df.empty:
        if "value" in df.columns[0]:
            reference = "value"
        elif "volume" in df.columns[0]:
            reference = "volume"
        else:
            return (None,) * 14  # 기준 컬럼이 없어 None 튜플 반환

        month_only = target_month.split(" ")[0]
        year_only = int(target_month.split(" ")[1])

        month_column = f"{reference}_{target_month}"
        month_yag_column = f"{reference}_{month_only} {year_only - 1}"
        month_2yag_column = f"{reference}_{month_only} {year_only - 2}"

        three_month_column = f"3mo_{reference}_{target_month}"
        three_month_yag_column = f"3mo_{reference}_{month_only} {year_only - 1}"

        MS_month_column = f"MS_{reference}_{target_month}"
        MS_month_yag_column = f"MS_{reference}_{month_only} {year_only - 1}"
        MS_month_2yag_column = f"MS_{reference}_{month_only} {year_only - 2}"

        MS_3month_column = f"3mo_MS_{reference}_{target_month}"
        MS_3month_yag_column = f"3mo_MS_{reference}_{month_only} {year_only - 1}"

        try:
            month_index = df.columns.get_loc(month_column)
            previous_month_index = month_index - 1 if month_index > 0 else None
            month_yag_index = df.columns.get_loc(month_yag_column)
            month_2yag_index = df.columns.get_loc(month_2yag_column)
            three_month_index = df.columns.get_loc(three_month_column)
            previous_three_month_index = three_month_index - 3 if three_month_index > 0 else None
            three_month_yag_index = df.columns.get_loc(three_month_yag_column)
            MS_month_index = df.columns.get_loc(MS_month_column)
            previous_MS_month_index = MS_month_index - 1 if MS_month_index > 0 else None
            MS_month_yag_index = df.columns.get_loc(MS_month_yag_column)
            MS_month_2yag_index = df.columns.get_loc(MS_month_2yag_column)
            MS_3month_index = df.columns.get_loc(MS_3month_column)
            previous_MS_3month_index = MS_3month_index - 3 if MS_3month_index > 0 else None
            MS_3month_yag_index = df.columns.get_loc(MS_3month_yag_column)
            return month_index, previous_month_index, month_yag_index, month_2yag_index, three_month_index, previous_three_month_index, three_month_yag_index, MS_month_index, previous_MS_month_index, MS_month_yag_index, MS_month_2yag_index, MS_3month_index, previous_MS_3month_index, MS_3month_yag_index
        except KeyError as e:
            st.warning(f"Cannot find column '{e}' corresponding to '{target_month}'.")
            return (None,) * 14
    else:
        return (None,) * 14

def monthly_performances(df):
    """DataFrame을 입력받아 다양한 성과 지표를 계산합니다."""
    item_index = df.columns.get_loc("ITEM")

    results = {}
    # 금액 성장률
    results['Value growth MoM'] = f"{((df.iloc[:, item_index+1] / df.iloc[:, item_index+2] * 100) - 100).round(1).iloc[0]} %"
    results['Monthly value growth YoY'] = f"{((df.iloc[:, item_index+1] / df.iloc[:, item_index+3] * 100) - 100).round(1).iloc[0]} %"
    results['Monthly value growth vs. 2 YAG'] = f"{((df.iloc[:, item_index+1] / df.iloc[:, item_index+4] * 100) - 100).round(1).iloc[0]} %"

    # 물량 성장률
    results['Volume growth MoM'] = f"{((df.iloc[:, item_index+15] / df.iloc[:, item_index+16] * 100) - 100).round(1).iloc[0]} %"
    results['Monthly volume growth YoY'] = f"{((df.iloc[:, item_index+15] / df.iloc[:, item_index+17] * 100) - 100).round(1).iloc[0]} %"
    results['Monthly volume growth vs. 2 YAG'] = f"{((df.iloc[:, item_index+15] / df.iloc[:, item_index+18] * 100) - 100).round(1).iloc[0]} %"

    # 금액 점유율 변화율
    results['Value share difference MoM'] = f"{(df.iloc[:, item_index+8] - df.iloc[:, item_index+9]).round(1).iloc[0]} %"
    results['Monthly value share difference vs. YAG'] = f"{(df.iloc[:, item_index+8] - df.iloc[:, item_index+10]).round(1).iloc[0]} %"
    results['Monthly value share difference vs. 2 YAG'] = f"{(df.iloc[:, item_index+8] - df.iloc[:, item_index+11]).round(1).iloc[0]} %"

    # 물량 점유율 변화율
    results['Volume share difference MoM'] = f"{(df.iloc[:, item_index+22] - df.iloc[:, item_index+23]).round(1).iloc[0]} %"
    results['Monthly volume share difference vs. YAG'] = f"{(df.iloc[:, item_index+22] - df.iloc[:, item_index+24]).round(1).iloc[0]} %"
    results['Monthly volume share difference vs. 2 YAG'] = f"{(df.iloc[:, item_index+22] - df.iloc[:, item_index+25]).round(1).iloc[0]} %"
    
    # kg당 단가 변화율
    results['Per kilo price difference MoM'] = f"{((pd.to_numeric(df.iloc[:, item_index+29].str.replace(',', '')) / pd.to_numeric(df.iloc[:, item_index+30].str.replace(',', '')) - 1) * 100).round(1).iloc[0]} %"
    results['Per kilo price difference vs. YAG'] = f"{((pd.to_numeric(df.iloc[:, item_index+29].str.replace(',', '')) / pd.to_numeric(df.iloc[:, item_index+31].str.replace(',', '')) - 1) * 100).round(1).iloc[0]} %"
    results['Per kilo price difference vs. 2 YAG'] = f"{((pd.to_numeric(df.iloc[:, item_index+29].str.replace(',', '')) / pd.to_numeric(df.iloc[:, item_index+32].str.replace(',', '')) - 1) * 100).round(1).iloc[0]} %"

    return pd.DataFrame([results]).T

def three_monthly_performances(df):
    """DataFrame을 입력받아 다양한 성과 지표를 계산합니다."""
    item_index = df.columns.get_loc("ITEM")

    results = {}

    # 금액 성장률
    results['3monthly_value growth vs. PP'] = f"{((df.iloc[:, item_index+5] / df.iloc[:, item_index+6] * 100) - 100).round(1).iloc[0]} %"
    results['3monthly_value growth YoY'] = f"{((df.iloc[:, item_index+5] / df.iloc[:, item_index+7] * 100) - 100).round(1).iloc[0]} %"

    # 물량 성장률
    results['3monthly_volume growth vs. PP'] = f"{((df.iloc[:, item_index+19] / df.iloc[:, item_index+20] * 100) - 100).round(1).iloc[0]} %"
    results['3monthly_volume growth YoY'] = f"{((df.iloc[:, item_index+19] / df.iloc[:, item_index+21] * 100) - 100).round(1).iloc[0]} %"

    # 금액 점유율 변화율
    results['3monthly_value share difference vs. PP'] = f"{(df.iloc[:, item_index+12] - df.iloc[:, item_index+13]).round(1).iloc[0]} %"
    results['3monthly_value share difference YoY'] = f"{(df.iloc[:, item_index+12] - df.iloc[:, item_index+14]).round(1).iloc[0]} %"

    # 물량 점유율 변화율
    results['3monthly_volume share difference vs. PP'] = f"{(df.iloc[:, item_index+26] - df.iloc[:, item_index+27]).round(1).iloc[0]} %"
    results['3monthly_volume share difference YoY'] = f"{(df.iloc[:, item_index+26] - df.iloc[:, item_index+28]).round(1).iloc[0]} %"

    # kg당 단가 변화율
    results['Per kilo 3monthly price difference vs. PP'] = f"{((pd.to_numeric(df.iloc[:, item_index+33].str.replace(',', '')) / pd.to_numeric(df.iloc[:, item_index+34].str.replace(',', '')) - 1) * 100).round(1).iloc[0]} %"
    results['Per kilo 3monthly price difference YoY'] = f"{((pd.to_numeric(df.iloc[:, item_index+33].str.replace(',', '')) / pd.to_numeric(df.iloc[:, item_index+35].str.replace(',', '')) - 1) * 100).round(1).iloc[0]} %"

    return pd.DataFrame([results]).T


def causal_analysis(df, timestamp=1):
    """
    Args:
        df (pd.DataFrame): 분석할 데이터프레임.
        timestamp (int, optional): 분석할 기간. 1: 월별, 2: 3개월별. Defaults to 1.

    Returns:
        pd.DataFrame: 분석 결과가 포함된 데이터프레임.
    """

    # 1. 대상 컬럼 필터링
    target_prefixes = ['value_', 'volume_', 'MS_value_', 'MS_volume_', 'price_'] if timestamp == 1 else ['3mo_value_', '3mo_volume_', '3mo_MS_value_', '3mo_MS_volume_', '3mo_price_']
    fact_columns = df.columns[:7]  # Markets, TOTAL R-T-E CEREAL, MANUFACTURER, TYPE, BRAND, SUBBRAND, ITEM
    data_columns = [col for col in df.columns if any(col.startswith(prefix) for prefix in target_prefixes)]
    filtered_df_col = list(fact_columns) + list(data_columns)
    filtered_df = df[filtered_df_col].copy() # copy warning 방지
    filtered_df_fact = df[fact_columns].copy() # copy warning 방지
    filtered_df_data = df[data_columns].copy() # copy warning 방지

    # 2. Extract unique timestamps from the remaining columns.
    postfix_set = [col.split('_')[-1] for col in data_columns]
    unique_timestamps = []
    seen = set()
    for item in postfix_set:
        if item not in seen:
            unique_timestamps.append(item)
            seen.add(item)

    # Ensure we have at least two timestamps to calculate differences.
    if len(unique_timestamps) < 2:
        print("Error: Need at least two time periods to calculate differences.")
        return filtered_df  # Return original DataFrame

    this_month = unique_timestamps[0]
    last_month = unique_timestamps[1]
    thismonth_lastyear = unique_timestamps[2] if len(unique_timestamps) > 2 else None # for 분기별

    # 3. Calculate differences for each prefix.
    for prefix in target_prefixes[0:2]:
        for i, col in enumerate(filtered_df.columns):
            if col == f'{prefix}{this_month}':
                this_month_index = i
            elif col == f'{prefix}{last_month}':
                last_month_index = i
            elif thismonth_lastyear and col == f'{prefix}{thismonth_lastyear}':
                thismonth_lastyear_index = i

        if 'this_month_index' in locals() and 'last_month_index' in locals():
            filtered_df_fact[f'{prefix}diff_{this_month}_{last_month}'] = ((filtered_df.iloc[:, this_month_index] / filtered_df.iloc[:, last_month_index] - 1) * 100).round(1)
        # if thismonth_lastyear and 'this_month_index' in locals() and 'thismonth_lastyear_index' in locals():
        if 'this_month_index' in locals() and 'thismonth_lastyear_index' in locals():
            filtered_df_fact[f'{prefix}diff_{this_month}_{thismonth_lastyear}'] = ((filtered_df.iloc[:, this_month_index] / filtered_df.iloc[:, thismonth_lastyear_index] - 1) * 100).round(1)
      
    for prefix in target_prefixes[2:5]:
        for i, col in enumerate(filtered_df.columns):
            if col == f'{prefix}{this_month}':
                this_month_index = i
            elif col == f'{prefix}{last_month}':
                last_month_index = i
            elif thismonth_lastyear and col == f'{prefix}{thismonth_lastyear}':
                thismonth_lastyear_index = i
        
        if 'this_month_index' in locals() and 'last_month_index' in locals():
            this_month_col = filtered_df.iloc[:, this_month_index]
            last_month_col = filtered_df.iloc[:, last_month_index]

            # 컬럼의 첫 번째 값의 타입을 확인하여 문자열인 경우에만 .str.replace 적용
            if isinstance(this_month_col.iloc[0], str):
                this_month_col = this_month_col.str.replace(',', '')
            if isinstance(last_month_col.iloc[0], str):
                last_month_col = last_month_col.str.replace(',', '')

            filtered_df_fact[f'{prefix}diff_{this_month}_{last_month}'] = (
                pd.to_numeric(this_month_col, errors='coerce') -
                pd.to_numeric(last_month_col, errors='coerce')
            ).round(1)

        if 'this_month_index' in locals() and 'thismonth_lastyear_index' in locals():
            this_month_col = filtered_df.iloc[:, this_month_index]
            last_year_col = filtered_df.iloc[:, thismonth_lastyear_index]

            # 컬럼의 첫 번째 값의 타입을 확인하여 문자열인 경우에만 .str.replace 적용
            if isinstance(this_month_col.iloc[0], str):
                this_month_col = this_month_col.str.replace(',', '')
            if isinstance(last_year_col.iloc[0], str):
                last_year_col = last_year_col.str.replace(',', '')

            filtered_df_fact[f'{prefix}diff_{this_month}_{thismonth_lastyear}'] = (
                pd.to_numeric(this_month_col, errors='coerce') -
                pd.to_numeric(last_year_col, errors='coerce')
            ).round(1)

    prefix = target_prefixes[4]
    for i, col in enumerate(filtered_df.columns):
        if col == f'{prefix}{this_month}':
            this_month_index = i
        elif col == f'{prefix}{last_month}':
            last_month_index = i
        elif thismonth_lastyear and col == f'{prefix}{thismonth_lastyear}':
            thismonth_lastyear_index = i

    if 'this_month_index' in locals() and 'last_month_index' in locals():
        this_month_col = filtered_df.iloc[:, this_month_index]
        last_month_col = filtered_df.iloc[:, last_month_index]

        this_month_col = this_month_col.str.replace(',', '')
        last_month_col = last_month_col.str.replace(',', '')

        filtered_df_fact[f'{prefix}diff_{this_month}_{last_month}'] = (
            (pd.to_numeric(this_month_col, errors='coerce') /
                pd.to_numeric(last_month_col, errors='coerce') - 1) * 100
            ).round(1)

    if 'this_month_index' in locals() and 'thismonth_lastyear_index' in locals():
        this_month_col = filtered_df.iloc[:, this_month_index]
        last_year_col = filtered_df.iloc[:, thismonth_lastyear_index]  
        
        this_month_col = this_month_col.str.replace(',', '')
        last_year_col = last_year_col.str.replace(',', '')

        filtered_df_fact[f'{prefix}diff_{this_month}_{thismonth_lastyear}'] = (
            (pd.to_numeric(this_month_col, errors='coerce') /
                pd.to_numeric(last_year_col, errors='coerce') - 1) * 100
        ).round(1)

    filtered_df_fact = filtered_df_fact.dropna(subset=[f'{prefix}diff_{this_month}_{thismonth_lastyear}'], axis=0)

    return filtered_df_fact


def monthly_diagnose_manufacturers(monthly_result_df, selected_manufacturer, selected_manufacturer_segment, latest_mo, month_ago):
    """
    선택한 제조사(Target)와 그 외 제조사(Source)의 데이터를 분석하여 진단 결과를 보여줍니다.

    Args:
        monthly_result_df (pd.DataFrame): 월별 실적 데이터프레임.
        selected_manufacturer (str): 분석 대상 제조사 이름.
        latest_mo (str): 가장 최근 월 정보.
        month_ago (str): 이전 월 정보.
    """

    if selected_manufacturer_segment is not None:
        target_df = monthly_result_df[monthly_result_df["MANUFACTURER"] == selected_manufacturer].copy()
        untarget_df = monthly_result_df[monthly_result_df["MANUFACTURER"] != selected_manufacturer].copy()

        segment_columns = [col for col in monthly_result_df.columns if col.startswith("SEGMENT")]
        if segment_columns:
            segment_filter_target = False
            segment_filter_untarget = True
            for segment_col in segment_columns:
                segment_filter_target = segment_filter_target | (target_df[segment_col] == selected_manufacturer_segment)
                segment_filter_untarget = segment_filter_untarget & (untarget_df[segment_col] != selected_manufacturer_segment)

            target_df = target_df[segment_filter_target]
            untarget_df = untarget_df[segment_filter_untarget]
        else:
            print("경고: 'SEGMENT'로 시작하는 컬럼이 데이터프레임에 없습니다.")
    else:
        target_df = monthly_result_df[monthly_result_df["MANUFACTURER"] == selected_manufacturer].copy()
        untarget_df = monthly_result_df[monthly_result_df["MANUFACTURER"] != selected_manufacturer].copy()
    diagnose_df = pd.DataFrame()
    sort_column = f"MS_value_diff_{latest_mo}_{month_ago}"

    # Target 제조사 진단
    if not target_df.empty:
        diagnose_target_df = target_df.T
        # st.write("check_point_func!!!", diagnose_target_df)
        diagnose_target_df.columns = ["Target"]
        diagnose_df = diagnose_df.join(diagnose_target_df, how="right")
        # st.subheader(f"Target Manufacturer_monthly: {selected_manufacturer}")
        # st.write(diagnose_df)
    else:
        st.warning(f"No data for the selected manufacturer '{selected_manufacturer}' available.")
        return  # Target 데이터 없으면 분석 중단

    # Untarget 제조사 진단
    if not untarget_df.empty:
        untarget_df = untarget_df[1:].copy()  # 첫 번째 행 제외

        if sort_column in target_df.columns and not target_df[sort_column].empty:
            target_ms_diff = float(target_df.iloc[0][sort_column])
            ascending_order = target_ms_diff > 0
            untarget_df_sorted = untarget_df.sort_values(by=sort_column, ascending=ascending_order).copy()

            share_sum = 0
            untarget_count = 0
            for i, share_point in enumerate(untarget_df_sorted[sort_column]):
                share_sum += float(share_point)
                if abs(share_sum) >= abs(target_ms_diff):
                    untarget_count = i + 1
                    break

            diagnose_untarget_transposed_df = untarget_df_sorted.head(untarget_count).T
            num_sources = diagnose_untarget_transposed_df.shape[1]
            new_columns = [f"Source{i+1}" for i in range(num_sources)]
            diagnose_untarget_transposed_df.columns = new_columns
            monthly_diagnose_df = diagnose_df.join(diagnose_untarget_transposed_df, how="right")
            return monthly_diagnose_df
            
        else:
            st.warning(f"Skip monthly diagnosis for the target manufacturer due to missing column '{sort_column}' or missing value.")
    else:
        st.info("No data for the causual manufacturer")


def three_monthly_diagnose_manufacturers(three_monthly_result_df, selected_manufacturer, selected_manufacturer_segment, latest_3mo, previous_3mo):
    """
    선택한 제조사(Target)와 그 외 제조사(Source)의 데이터를 분석하여 진단 결과를 보여줍니다.

    Args:
        monthly_result_df (pd.DataFrame): 월별 실적 데이터프레임.
        selected_manufacturer (str): 분석 대상 제조사 이름.
        latest_mo (str): 가장 최근 월 정보.
        month_ago (str): 이전 월 정보.
    """

    if selected_manufacturer_segment is not None:
        target_df = three_monthly_result_df[three_monthly_result_df["MANUFACTURER"] == selected_manufacturer].copy()
        untarget_df = three_monthly_result_df[three_monthly_result_df["MANUFACTURER"] != selected_manufacturer].copy()

        segment_columns = [col for col in three_monthly_result_df.columns if col.startswith("SEGMENT")]
        if segment_columns:
            segment_filter_target = False
            segment_filter_untarget = True
            for segment_col in segment_columns:
                segment_filter_target = segment_filter_target | (target_df[segment_col] == selected_manufacturer_segment)
                segment_filter_untarget = segment_filter_untarget & (untarget_df[segment_col] != selected_manufacturer_segment)

            target_df = target_df[segment_filter_target]
            untarget_df = untarget_df[segment_filter_untarget]
        else:
            print("경고: 'SEGMENT'로 시작하는 컬럼이 데이터프레임에 없습니다.")
    else:
        target_df = three_monthly_result_df[three_monthly_result_df["MANUFACTURER"] == selected_manufacturer].copy()
        untarget_df = three_monthly_result_df[three_monthly_result_df["MANUFACTURER"] != selected_manufacturer].copy()
    diagnose_df = pd.DataFrame()
    sort_column = f"3mo_MS_value_diff_{latest_3mo}_{previous_3mo}"

    # Target 제조사 진단
    if not target_df.empty:
        diagnose_target_df = target_df.T
        diagnose_target_df.columns = ["Target"]
        diagnose_df = diagnose_df.join(diagnose_target_df, how="right")
        # st.subheader(f"Target Manufacturer_3monthly: {selected_manufacturer}")
        # st.write(diagnose_df)
    else:
        st.warning(f"No data for the selected manufacturer '{selected_manufacturer}' available.")
        return  # Target 데이터 없으면 분석 중단
    
    # Untarget 제조사 진단
    if not untarget_df.empty:
        untarget_df = untarget_df[1:].copy()  # 첫 번째 행 제외

        if sort_column in target_df.columns and not target_df[sort_column].empty:
            target_ms_diff = float(target_df.iloc[0][sort_column])
            ascending_order = target_ms_diff > 0
            untarget_df_sorted = untarget_df.sort_values(by=sort_column, ascending=ascending_order).copy()

            share_sum = 0
            untarget_count = 0
            for i, share_point in enumerate(untarget_df_sorted[sort_column]):
                share_sum += float(share_point)
                if abs(share_sum) >= abs(target_ms_diff):
                    untarget_count = i + 1
                    break

            diagnose_untarget_transposed_df = untarget_df_sorted.head(untarget_count).T

            num_sources = diagnose_untarget_transposed_df.shape[1]
            new_columns = [f"Source{i+1}" for i in range(num_sources)]
            diagnose_untarget_transposed_df.columns = new_columns

            three_monthly_diagnose_df = diagnose_df.join(diagnose_untarget_transposed_df, how="right")
            return three_monthly_diagnose_df
        else:
            st.warning(f"Skip monthly diagnosis for the target manufacturer due to missing column '{sort_column}' or missing value.")
    else:
        st.info("No data for the causual manufacturer")