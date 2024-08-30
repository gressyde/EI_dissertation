#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 12:08:21 2024

@author: vladelec
"""
import pandas as pd
import xlwings as xw

def _process(filename):
    try:
        app = xw.App(visible=False)  # Open Excel application in the background
        wb = app.books.open(filename)
        sheet = wb.sheets['WT_Sectors']  # Replace with your sheet name if different
        
        # Read data into DataFrame
        df = sheet.used_range.options(pd.DataFrame, index=False, header=True).value
        
        # Unprotect the sheet if it is protected
        if sheet.protection:
            sheet.protection = False  # This should unprotect the sheet if it is protected
        
        wb.save()
        wb.close()
        app.quit()
        return df
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

filename = '/Users/vladelec/Desktop/Exeter /Summer Project/FTT_StandAlone-main/2024-01-30 - 2H 2023 LCOE Data Viewer Tool copy.xlsb'
df = _process(filename)
