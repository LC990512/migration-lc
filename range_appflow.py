import pandas as pd
import numpy as np
import sys
import math
import re
import ast
import os
from lxml import etree
import time
from functools import cmp_to_key

start_time = time.time()

#寻找statename时，打不开文件直接return了，错误
num = 0
pro_cnt = 0
data = pd.DataFrame()
problem = set()

#按相对位置定位
def search_rel(node, search_type, id, re_xpath):
    if search_type == 'id':
        if node.get('resource-id') == id:
            return(node.get('bounds'))
    elif search_type == 'xpath':
        if node.get('class') == re_xpath['class'] and node.get('content-desc') == re_xpath['cd']:
            return(node.get('bounds'))
    if len(node) > 0:
        for child in node:
            bounds = search_rel(child, search_type, id, re_xpath)
            if bounds: return bounds
    return None

#按绝对位置定位
def search_abs(root, xpath):
    bounds = ''
    find_all_node = etree.XPath(xpath)
    nodelist = find_all_node(root)
    if len(nodelist) != 0:
        if len(nodelist) > 1:
            print("more than one node has the same xpath in one .uix")
            print("state_path", xpath)
        if nodelist[0].get("bounds") != "":
            bounds = nodelist[0].get("bounds")
    return bounds

#改写xpath
def revise_xpath(ori_xpath):
    # absolute path
    element_list = ori_xpath.split("/")
    revised_xpath = '/hierarchy'
    for index in range(2,len(element_list)):
        element = element_list[index]
        if "[" in element:
            xpath_element = "/node[@class='"+element.split("[")[0]+"']"+"["+element.split("[")[1]
        else:
            xpath_element = "/node[@class='"+element+"']"
        revised_xpath = revised_xpath + xpath_element
    return revised_xpath

def get_location(index, app, id, xpath):
    #sta = etsy/sign_in_or_sign_up.uix
    sta = ''
    bounds = ''
    re_xpath = {'class':"", 'cd':""}    
    xpath_type = ''
    search_type = ''
    cnt = 0
        
    if (str(id) == 'nan') and (str(xpath) == 'nan'):
        return sta, bounds, cnt
    
    #先从GT_etsy_revise.csv中找sta
    df_tar_path = 'GT_revise/'+'GT_' + app + '_revise.csv'
    df_tar = pd.read_csv('GT_revise/'+'GT_' + app + '_revise.csv')
    sta_list = []
    
    
    for i in df_tar.index:
        #ori_id // ori_appium_xpath
        sta = ''
        if id == df_tar.loc[i]['ori_id']:
            if (xpath == 'nan'):
                search_type = 'id'
                sta = df_tar.loc[i]['state_name']
            if  (xpath == df_tar.loc[i]['ori_appium_xpath']):
                search_type = 'both'
                sta = df_tar.loc[i]['state_name']  
        elif id == 'nan':
            if xpath == df_tar.loc[i]['ori_appium_xpath']:
                search_type = 'xpath'
                sta = df_tar.loc[i]['state_name']   
        if sta:
            if str(sta) != 'nan':
                sta = sta.replace(' ', '\t')
                p = sta.split('\t')
                for item in p:
                    sta_list.append(item)


    #改写xpath
    if '/hierarchy' in xpath:
        xpath = revise_xpath(xpath)
        xpath_type = 'absolute'
    else:
        if (str(xpath) != 'nan'):
            pattern = re.compile(r'\/\/(.*?)\[')
            if (pattern.findall(xpath)):
                re_xpath['class'] = pattern.findall(xpath)[0]
            pattern = re.compile(r'\"(.*?)\"')
            if (pattern.findall(xpath)):
                re_xpath['cd'] = pattern.findall(xpath)[0]
        xpath_type = 'relative'

    
    #遍历不同的xml处理无sta的情况
    flag = 0
    if app == 'fivemiles':
        app = '5miles'
    mydir = 'screenshots/'+app
    if len(sta_list) == 0:
        if (str(xpath) == 'nan'):
            search_type = 'id'
        elif (str(id) == 'nan'):
            search_type = 'xpath'
        else:
            search_type = 'both'
        
        for root, dirs, files in os.walk(mydir):
            for file in files:
                if file.endswith(".xml") or file.endswith(".uix"):
                    targetFilePath = app+'/'+ file
                    sta_list.append(targetFilePath)
                    flag = 1


    #无文件
    map = []
    map_bounds = []
    for path_item in sta_list:
        flag_abs = 0
        try:
            tree = etree.parse('screenshots/'+path_item)
        except:
            i = sta_list.index(path_item)
            continue
            #return sta, bounds, cnt
        root = tree.getroot()
        
        #先按相对位置找
        for node in root:
            if search_type == 'both':
                bounds_item = search_rel(node, 'id', id, re_xpath)
                if not bounds_item:
                    bounds_item = search_rel(node, 'xpath', id, re_xpath)
            else:
                bounds_item = search_rel(node, search_type, id, re_xpath)
            if bounds_item:
                flag_abs = 1
                
                map.append(path_item)
                map_bounds.append(bounds_item)
                cnt = cnt + 1
        
      
        #没找到再按绝对位置找
        if not flag_abs:
            if (xpath_type == 'absolute') and (search_type != 'id'):
                bounds_item = search_abs(root, xpath)
                if bounds_item:
                    sta = path_item
                    map.append(path_item)
                    map_bounds.append(bounds_item)
                    cnt = cnt + 1

  
    if len(map) == 0:
        return sta, bounds, cnt
    
    #uix xml去重 5miles/sign_in.uix
    map_choose = []
    map_bounds_choose = []
    for i in range(0, len(map)):
        p = map[i].split('.')[0]
        if p not in map_choose:
            map_choose.append(p)
            map_bounds_choose.append(map_bounds[i])
        else:
            #存在uix和xml位置不一样的情况
            index_bounds = map_choose.index(p)
            if (map_bounds_choose[index_bounds] != map_bounds[i]):
                map_choose.append(p)
                map_bounds_choose.append(map_bounds[i])
               
    if len(map_choose) > 1:
        sta = str(map_choose)
        bounds = str(map_bounds_choose)
    else:
        sta = map_choose[0]
        bounds = map_bounds_choose[0]

 
    return sta, bounds, len(map_choose)
    

#是否在同一界面
def get_screen_bool(i, j, x):
    #xpath = null
    
    if (x.iloc[i]['state_name']):
        if (x.iloc[i]['state_name'] == x.iloc[j]['state_name']):
            return True    
    return False

#寻找以srcapp分组的最长子序列
def lcs(y):
    dict = {}
    order = {}
    flag = step = 0
    first_src_index = y['src_index'].min()
    y.sort_values(by="src_index", inplace=True, ascending=True) 
    
    for i in y.index:
        if ((y.loc[i]['ori_predict_label'] == 1) and (y.loc[i]['label'] == 1)) or (str(y.loc[i]['evaluate_ori']) == 'tn'):
            flag_exist = 0
            if (y.loc[i]['src_index'] == first_src_index):
                flag = 1
                for key, value in dict.items():
                    if get_same_bool(y.loc[i]['index'], value, y):
                        flag_exist = 1
                if flag_exist == 0:
                    if str(y.loc[i]['correct_tgt_index']) != 'nan':
                        dict[y.loc[i]['correct_tgt_index']] = y.loc[i]['index']
                        if (y.loc[i]['state_name'] not in order.values()):
                            order[step] = y.loc[i]['state_name']
                            step += 1

            elif flag == 1:
                for key, value in dict.items():
                    if get_same_bool(y.loc[i]['index'], value, y):
                        flag_exist = 1
                if flag_exist == 0:
                    if str(y.loc[i]['correct_tgt_index']) != 'nan':
                        dict[y.loc[i]['correct_tgt_index']] = y.loc[i]['index']
                        if (y.loc[i]['state_name'] not in order.values()):
                            order[step] = y.loc[i]['state_name']
                            step += 1
        else:
            break
    y['step'] = len(dict) 
    y['lcs'] = str(dict)
    y['order'] = str(order)
    return y

def get_same_pos(i, j, x):
    sta_dict_i = []
    bounds_dict_i = []
    sta_dict_j = []
    bounds_dict_j = []
    flag = 1
    if '[' in x.loc[x['index']==i, 'state_name'].values[0]:
        sta_dict_i = ast.literal_eval(x.loc[x['index']==i, 'state_name'].values[0])
        bounds_dict_i = ast.literal_eval(x.loc[x['index']==i, 'bounds'].values[0])
    if '[' in x.loc[x['index']==j, 'state_name'].values[0]:
        sta_dict_j = ast.literal_eval(x.loc[x['index']==j, 'state_name'].values[0])
        bounds_dict_j = ast.literal_eval(x.loc[x['index']==j, 'bounds'].values[0]) 
    if (len(sta_dict_i) == 0):
        if x.loc[x['index']==i, 'state_name'].values[0]:
            sta_dict_i.append(x.loc[x['index']==i, 'state_name'].values[0])
            bounds_dict_i.append(x.loc[x['index']==i, 'bounds'].values[0])
    if (len(sta_dict_j) == 0):
        if x.loc[x['index']==j, 'state_name'].values[0]:
            sta_dict_j.append(x.loc[x['index']==j, 'state_name'].values[0])
            bounds_dict_j.append(x.loc[x['index']==j, 'bounds'].values[0])

    if len(sta_dict_i) > len(sta_dict_j):
        i, j = j, i
        sta_dict_i, sta_dict_j = sta_dict_j, sta_dict_i
        bounds_dict_i, bounds_dict_j = bounds_dict_j, bounds_dict_i
    
    flag_in = 0
    if len(sta_dict_i) > 0:
        if len(sta_dict_j) > 0:
            for k in range(0, len(sta_dict_i)):
                if sta_dict_i[k] in sta_dict_j:
                    #index_j = sta_dict_j.index(sta_dict_i[k])
                    indices = [idx for idx, val in enumerate(sta_dict_j) if val == sta_dict_i[k]]
                    flag_in = 1
                    exist = 0
                    for index_j in range(0, len(indices)):
                        if bounds_dict_i[k] == bounds_dict_j[indices[index_j]]:
                            exist = 1
                    if (exist == 0):
                        return False
            if flag_in == 0:
                return False   
        elif not x.loc[x['index']==j, 'state_name'].values[0]:
            return False
        elif x.loc[x['index']==j, 'state_name'].values[0] in sta_dict_i:
            if x.loc[x['index']==j, 'bounds'].values[0] != bounds_dict_i[sta_dict_i.index(x.loc[x['index']==j, 'state_name'].values[0])]:
                return False
        elif x.loc[x['index']==j, 'state_name'].values[0] not in sta_dict_i:
            return False
    elif len(sta_dict_j) > 0:
        if not x.loc[x['index']==i, 'state_name'].values[0]:
            return False
        elif x.loc[x['index']==i, 'state_name'].values[0] in sta_dict_j:
            if x.loc[x['index']==i, 'bounds'].values[0] != bounds_dict_j[sta_dict_j.index(x.loc[x['index']==i, 'state_name'].values[0])]:
                return False
        elif x.loc[x['index']==i, 'state_name'].values[0] not in sta_dict_j:
            return False
    else:
        if (str(x.loc[x['index']==i, 'state_name'].values[0]) != str(x.loc[x['index']==j, 'state_name'].values[0])):
            return False
        elif pd.isna(x.loc[x['index']==i, 'state_name'].values[0]):
            return False
        elif (str(x.loc[x['index']==i, 'pos'].values[0]) != str(x.loc[x['index']==j, 'pos'].values[0])):
            return False
    return True


def get_same_bool(i, j, x):
    #predict_input canonical
    special = 'clear_and_send_keys_and_hide_keyboard'
    if (str(x.loc[x['index']==i, 'tgt_text'].values[0]) != str(x.loc[x['index']==j, 'tgt_text'].values[0])):
        return False
    if (str(x.loc[x['index']==i, 'tgt_content'].values[0]) != str(x.loc[x['index']==j, 'tgt_content'].values[0])):
        return False
    if (str(x.loc[x['index']==i, 'predict_action'].values[0]) != str(x.loc[x['index']==j, 'predict_action'].values[0])):
        if (str(x.loc[x['index']==i, 'predict_action'].values[0]) != special) and (str(x.loc[x['index']==j, 'predict_action'].values[0]) != special):
            return False
    if not get_same_pos(i, j, x):
        return False
    if  str(x.loc[x['index']==i, 'tgt_index'].values[0]) == 'nan' and str(x.loc[x['index']==i, 'tgt_content'].values[0]) == 'nan' and str(x.loc[x['index']==i, 'predict_action'].values[0]) == 'nan':
        return False
    return True

def get_empty_bool(i, x):
    #Tgt_add_id and tgt_add_xpath and tgt_text and tgt_content 都为空的时候不给tgt_index 
    if (str(x.loc[x['index']==i, 'tgt_add_id'].values[0]) == 'nan') and (str(x.loc[x['index']==i, 'tgt_add_xpath'].values[0]) == 'nan') and (str(x.loc[x['index']==i, 'tgt_text'].values[0]) == 'nan') and (str(x.loc[x['index']==i, 'tgt_content'].values[0]) == 'nan'):
        return True
    return False

#type = 1 single
#type = 2 muti
def single_state_fill(x, dict, dict_max, type):
    for i in range(len(x)-1, -1, -1):
        #填过的 / 找不到sta / 没有order / fn / empty的跳过
        if str(x.iloc[i]['tgt_index']) != 'nan':
            continue
        if (not x.iloc[i]['state_name']) or (x.iloc[i]['state_order'] == sys.maxsize):
            continue     
        if (str(x.iloc[i]['evaluate_ori']) == 'fn') or get_empty_bool(x.iloc[i]['index'], x):
            continue

        
        if (type == 1):
            if x.iloc[i]['cnt'] > 1:
                continue

        #判断是否已有该组件
        flag = 0
        for key, value in dict.items():
            if get_same_bool(x.iloc[i]['index'], value, x):
                x.iloc[i, 22] = key
                flag = 1
        if (flag == 1):
            continue

        #寻找左右区间，第一个有值且在同一界面内的
        l = i  
        r = i
        while ((str(x.iloc[l]['tgt_index']) == 'nan')):
            if (l - 1 < 0):
                break
            l = l - 1
        while ((str(x.iloc[r]['tgt_index']) == 'nan')):
            if (r + 1 == len(x)):
                break
            r = r + 1
        
        if (str(x.iloc[l]['tgt_index']) == 'nan'):
            #只有右侧
            if (str(x.iloc[r]['tgt_index']) != 'nan'):
                index = x.iloc[r]['tgt_index'] - 1
                while ((index in dict.keys()) or ((index > -3) and (index < 0))):
                    #如果是同一组件
                    if index in dict.keys():
                        if get_same_bool(x.iloc[i]['index'], dict[index], x):
                            break
                    index = index - 1
                x.iloc[i, 22] = index
                dict[index] = x.iloc[i]['index']    
        else:
            if get_screen_bool(i, l, x) or not get_screen_bool(i, l, x):
                #左右都有
                if ((str(x.iloc[r]['tgt_index']) != 'nan')):
                    #左右相等
                    if get_same_bool(x.iloc[l]['index'], x.iloc[r]['index'], x):
                        index = x.iloc[r]['tgt_index'] - 0.1
                        while ((index in dict.keys()) or ((index > -3) and (index < 0))):
                            index = index - 0.1
                    else:    
                        index = (x.iloc[l]['tgt_index'] + x.iloc[r]['tgt_index']) / 2   
                        while (index in dict.keys()):
                            index = (x.iloc[l]['tgt_index'] + index) / 2
                    x.iloc[i, 22] = index
                    dict[index] = x.iloc[i]['index']    
                #只有左侧
                else:
                    index = x.iloc[l]['tgt_index'] + 1
                    while ((index in dict.keys()) or ((index > dict_max) and (index < 20))):
                        if index in dict.keys():
                            if get_same_bool(x.iloc[i]['index'], dict[index], x):
                                break
                        index = index + 1
                    x.iloc[i, 22] = index
                    dict[index] = x.iloc[i]['index']
    return x, dict

def no_state_or_order_fill(x, dict):
    for i in range(0, len(x)):
        if str(x.iloc[i]['tgt_index']) != 'nan':
            continue
        if ((str(x.iloc[i]['type']) == 'EMPTY_EVENT') or (str(x.iloc[i]['type']) == 'SYS_EVENT')) :
            continue
        if (str(x.iloc[i]['evaluate_ori']) == 'fn') or get_empty_bool(x.iloc[i]['index'], x):
            continue
            
        if (not x.iloc[i]['state_name'] or x.iloc[i]['state_order'] == sys.maxsize):
            flag = 0
            for key, value in dict.items():
                if get_same_bool(x.iloc[i]['index'], value, x):
                    x.iloc[i, 22] = key
                    flag = 1
            if (flag == 1):
                continue
            # y = x[x['src_app'] == x.iloc[i]['src_app']]
            # y.sort_values(by=['index'], inplace=True, ascending=True) 
            index =  max(max(dict) + 1, 20)
            x.iloc[i, 22] = index
            dict[index] = x.iloc[i]['index']      
    return x, dict                        

def assign_state_order_improved(x):
    tgt_app = ''
    function = ''
    
    result_x = x.copy()
    for i in x.index:
        if (str(x.loc[i]['tgt_index']) != 'nan') or (get_empty_bool(x.loc[i]['index'], x)):
            continue
        if str(x.loc[i]['evaluate_ori']) == 'fn':
            continue
        if (x.loc[i]['state_name']) and (x.loc[i]['state_order'] == sys.maxsize):
            state = x.loc[i]['state_name']
            relevant_rows = x[x['state_name'] == state]['src_app'].unique()
            order_ranges = []

            for app in relevant_rows:
                # 获取当前 src_app 中的所有状态及其顺序
                app_states = x[x['src_app'] == app]#.sort_values(by='state_order', na_position='last')
                app_states = app_states[~app_states.apply(lambda y: get_empty_bool(y['index'], x) or str(y['evaluate_ori']) == 'fn', axis=1)]

                 
                if state not in app_states['state_name'].values:
                    continue

                # 确定当前 state_name 在该 src_app 中的位置
                state_index = app_states[app_states['state_name'] == state].index[0]
                prev_state_order = app_states.loc[:state_index, 'state_order'].replace(sys.maxsize, np.nan).dropna().max()
                next_state_order = app_states.loc[state_index:, 'state_order'].replace(sys.maxsize, np.nan).dropna().min()

                # 确定范围
                if pd.isna(prev_state_order) and pd.isna(next_state_order):
                    continue
                elif pd.isna(prev_state_order):
                    order_ranges.append((min(next_state_order-1, 0), next_state_order))
                elif pd.isna(next_state_order):
                    order_ranges.append((prev_state_order, app_states['state_order'].replace(sys.maxsize, np.nan).dropna().max() + 1))
                else:
                    order_ranges.append((prev_state_order, next_state_order))
            
            chosen_order = sys.maxsize
            if order_ranges:
                valid_range = order_ranges[0]
                for r in order_ranges[1:]:
                    valid_range_i = (max(valid_range[0], r[0]), min(valid_range[1], r[1]))
                    if valid_range_i[0] >= valid_range_i[1]:  # 如果交集为空，停止查找
                        break
                    else:
                        valid_range = valid_range_i

                # 如果找到有效的交集范围，则选择范围内的任意值
                if valid_range[0] < valid_range[1]:
                    chosen_order = (valid_range[0] + valid_range[1]) / 2
                
            else:
                chosen_order = 1

            result_x.loc[result_x['state_name'] == state, 'state_order'] = chosen_order
            x.loc[x['state_name'] == state, 'state_order'] = chosen_order
            
    return result_x

def set_order(x):
    #(not testSignUp): main/sign_in_or_sign_up/start-> sign_in-> other-> sign_up 
    #(testSignUp):     main/sign_in_or_sign_up/start-> sign_up-> sign_in-> other

    if x.iloc[0]['function'] != 'testSignUp':
        for i in x.index:
            if (not x.loc[i]['state_name']):
                continue
            name = x.loc[i]['state_name'].split('/')[1]

            if name == 'main' or name == 'sign_in_or_sign_up' or name == 'start':
                x.loc[i, 'state_order'] = 1
            elif name == 'sign_in':
                x.loc[i, 'state_order'] = 2
            elif name == 'sign_up':
                x.loc[i, 'state_order'] = 4
            else:
                x.loc[i, 'state_order'] = 3
    else:
        for i in x.index:
            if (not x.loc[i]['state_name']):
                continue
            name = x.loc[i]['state_name'].split('/')[1]
            if name == 'main' or name == 'sign_in_or_sign_up' or name == 'start':
                x.loc[i, 'state_order'] = 1
            elif name == 'sign_up':
                x.loc[i, 'state_order'] = 2
            elif name == 'sign_in':
                x.loc[i, 'state_order'] = 3
            else:
                x.loc[i, 'state_order'] = 4
    return x                

def fill_pos(x, dict):
    tgt_app = 'wish'
    function = 'testSignUp'
    pmin = 100
    pmax = -100
    cnt_no_state_order = 0
    
    #只有一个src_app 按照现有顺序直接赋值
    if len(x.groupby(['src_app'], group_keys=True).groups) == 1:
        for i in x.index:
            if str(x.loc[i]['tgt_index']) != 'nan':
                continue
            if get_empty_bool(x.loc[i]['index'], x):
                continue   
            if str(x.loc[i]['evaluate_ori']) == 'fn':
                continue
            if str(x.loc[i]['ori_tgt_index']) != 'nan':
                x.loc[i, 'tgt_index'] = x.loc[i]['ori_tgt_index']
                dict[x.loc[i]['ori_tgt_index']] = x.loc[i]['index']    
        return x


    #如果不存在最长可执行子序列                
    if len(dict) == 0:
        flag_tgt_index = 0
        for i in x.index:
            if str(x.loc[i, 'tgt_index']) != 'nan':
                flag_tgt_index = 1
                break
        if flag_tgt_index == 0:
            for i in x.index:
                if str(x.loc[i]['ori_tgt_index']) != 'nan':
                    x.loc[i, 'tgt_index'] = x.loc[i]['ori_tgt_index']
                    x.loc[i, 'state_order'] = 1
                    value = x.loc[i]['state_name']
                    first_value = x.loc[i]['index']  
                    dict[x.loc[i]['ori_tgt_index']] = x.loc[i]['index']    
                    break
            for i in x.index:
                if (str(x.loc[i]['evaluate_ori']) == 'fn') or get_empty_bool(x.loc[i]['index'], x):
                    continue
                if (value == x.loc[i]['state_name']):
                    if (x.loc[i]['state_name']) or ((not x.loc[i]['state_name']) and get_same_bool(x.loc[i]['index'], first_value, x)):
                        x.loc[i, 'state_order'] = 1
    
    if len(dict) > 0:
        dict_max = max(dict)
    else:
        return x    
    
    #有无无法判断的界面顺序
    for i in x.index:
        if (str(x.loc[i]['tgt_index']) != 'nan') or (get_empty_bool(x.loc[i]['index'], x)):
            continue
        if str(x.loc[i]['evaluate_ori']) == 'fn':
            continue
    
        if (x.loc[i]['state_name']) and (x.loc[i]['state_order'] == sys.maxsize):
            cnt_no_state_order = cnt_no_state_order + 1

  
    if cnt_no_state_order > 0:
        x = assign_state_order_improved(x)

    # x = set_order(x) #指定界面分类
    x.sort_values(by=['state_order', 'state_name', 'by', 'bx', 'src_app', 'src_index', 'index'], inplace=True, ascending=True) 
        
    x, dict = single_state_fill(x, dict, dict_max, 1)
    
    x.sort_values(by=['index'], inplace=True, ascending=True) 
    x, dict = no_state_or_order_fill(x, dict)
    
    return x


def muti_state(x):
    m = 0
    #定位所需列
    state_name_column = x.columns.get_loc("state_name")
    bounds_column = x.columns.get_loc("bounds")
    cnt_column = x.columns.get_loc("cnt")
    bx_column = x.columns.get_loc("bx")
    by_column = x.columns.get_loc("by")
    pos_column = x.columns.get_loc("pos")

    for i in range(0, len(x)):
        flag = 1
        if x.iloc[i]['cnt'] > 1:
            flag = 2
            sta_dict = ast.literal_eval(x.iloc[i]['state_name'])
            bounds_dict = ast.literal_eval(x.iloc[i]['bounds'])
        
            l = i
            flagl = 0
            while l > 0:
                l = l - 1
                if (l < 0):
                    break
                if (x.iloc[l]['src_app'] != x.iloc[i]['src_app']):
                    break
                if (x.iloc[l]['state_name'] in sta_dict):
                    flagl = 1
                    break
            
            r = i 
            flagr = 0
            while r < len(x)-1:
                r = r + 1
                if (x.iloc[r]['src_app'] != x.iloc[i]['src_app']):
                    break
                if (x.iloc[r]['state_name'] in sta_dict):
                    flagr = 1
                    break
            
            if flagl:
                x.iloc[i, state_name_column] = x.iloc[l]['state_name']
                if (flagr) and (r-i < i-l):
                    x.iloc[i, state_name_column] = x.iloc[r]['state_name']
                x.iloc[i, bounds_column] = bounds_dict[sta_dict.index(x.iloc[i]['state_name'])]
                flag = 1
                m = m +1
            elif flagr:
                x.iloc[i, state_name_column] = x.iloc[r]['state_name']
                x.iloc[i, bounds_column] = bounds_dict[sta_dict.index(x.iloc[i]['state_name'])]
                flag = 1
                m = m +1
        if flag == 1:
            pattern = re.compile(r'\[(.*?)\]')
            locs = pattern.findall(x.iloc[i]['bounds'])
            if (locs):
                x1, y1 = int(locs[0].split(',')[0]), int(locs[0].split(',')[1])
                x2, y2 = int(locs[1].split(',')[0]), int(locs[1].split(',')[1])
                x.iloc[i, bx_column] = round((x1+x2)/2, 2)
                x.iloc[i, by_column] = round((y1+y2)/2, 2)
            x.iloc[i, cnt_column] = 1  #cnt
            x.iloc[i, pos_column] = str(round(x.iloc[i]['bx'], 2)) + ' ' + str(round(x.iloc[i]['by'], 2))
    
    for i in x.index:
        if x.loc[i]['cnt'] > 1:
            for j in x.index:
                if x.loc[j]['cnt'] == 1:
                    if get_same_bool(x.loc[i]['index'], x.loc[j]['index'], x):
                        m = m + 1
                        x.loc[i, 'cnt'] = 1
                        x.loc[i, 'state_name'] = x.loc[j]['state_name']
                        x.loc[i, 'bounds'] = x.loc[j]['bounds']
                        x.loc[i, 'bx'] = x.loc[j]['bx']
                        x.loc[i, 'by'] = x.loc[j]['by']
                        x.loc[i, 'pos'] = x.loc[j]['pos']
    return m
 
def solve(x):
    #标记相对位置
    global num
    num = num + 1
    for i in x.index:
        x.loc[i, 'state_name'], x.loc[i, 'bounds'], x.loc[i, 'cnt'] = get_location(x.loc[i]['index'], x.loc[i]['tgt_app'], str(x.loc[i]['tgt_add_id']), str(x.loc[i]['tgt_add_xpath']))
        x.loc[i, 'bx'] = sys.maxsize
        x.loc[i, 'by'] = sys.maxsize
        x.loc[i, 'pos'] = str(sys.maxsize) + ' ' + str(sys.maxsize)
        x.loc[i, 'list'] = x.loc[i, 'state_name']   
    x.sort_values(by=['src_app', 'index'], inplace=True, ascending=True) 

    #处理多statename情况
    while True:        
        if muti_state(x) == 0:
            break


    #获取最长子序列并记录在字段lcs中
    result_group_src= x.groupby(['src_app'], group_keys=True).apply(lcs)
    lcs_idx = result_group_src['step'].argmax()
    lcs_max = result_group_src['step'].max()
    
    lcs_dict = {}
    order_dict = {}
    for i in x.index:
        x.loc[i, 'state_order'] = sys.maxsize
    
    if (lcs_max > 0):      
        lcs_dict = ast.literal_eval(result_group_src.iloc[lcs_idx]['lcs'])
        order_dict = ast.literal_eval(result_group_src.iloc[lcs_idx]['order'])
        #按照最长子序列填充tgt_index + 记录界面相对位置
        for i in x.index:
            if (str(x.loc[i]['evaluate_ori']) == 'fn') or get_empty_bool(x.loc[i]['index'], x):
                continue
            for key, value in lcs_dict.items():
                if get_same_bool(x.loc[i]['index'], value, x):
                    x.loc[i, 'tgt_index'] = key
                    break
            position = 0
            for key, value in order_dict.items():
                if (value == x.loc[i]['state_name']):
                    first_value = list(lcs_dict.values())[position]
                    if ((not x.loc[i]['state_name']) and (get_same_bool(x.loc[i]['index'], first_value, x))) or (x.loc[i]['state_name']):
                        x.loc[i, 'state_order'] = key
                position += 1
    
    #2.按照相对位置填充tgt_index
    x = fill_pos(x, lcs_dict)

    #检查
    return x

def main():
    df = pd.read_csv('data.csv')
    #组内找相对位置+最长子序列
    result_group = df.groupby(['tgt_app', 'function'], group_keys=True).apply(solve)
    #反向填充原始文件
    for i in range(0, len(result_group)):
        index = result_group.iloc[i]['index']
        df.loc[df['index']==index, 'tgt_index'] = result_group.iloc[i]['tgt_index']
        df.loc[df['index']==index, 'state_name'] = result_group.iloc[i]['state_name']
        df.loc[df['index']==index, 'state_order'] = result_group.iloc[i]['state_order']
        df.loc[df['index']==index, 'pos'] = result_group.iloc[i]['pos']
        df.loc[df['index']==index, 'list'] = result_group.iloc[i]['list']
    
    selected_columns_df = df[['index', 'src_app', 'tgt_app', 'function', 'canonical', 'tgt_add_id', 'tgt_add_xpath', 'ori_tgt_index', 'ori_predict_label', 'label', 'src_index','correct_tgt_index', 'tgt_index', 'state_name', 'state_order', 'pos']]
    selected_columns_df.to_csv("result_appflow.csv", index=False)
    
    

if __name__ == "__main__":
    main()

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Total execution time: {elapsed_time:.2f} seconds")
