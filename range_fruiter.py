import pandas as pd
import numpy as np
import sys
import math
import re
import ast
import os
from lxml import etree
from functools import cmp_to_key

#寻找statename时，打不开文件直接return了，错误
num = 0
pro_cnt = 0
data = pd.DataFrame()
problem = []

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
    
    
    #print("get location1:", index, app)
    
    if (str(id) == 'nan') and (str(xpath) == 'nan'):
        #print("no id, xpath:", app, id, xpath)
        return sta, bounds, cnt
    
    #先从GT_etsy_revise.csv中找sta
    df_tar_path = 'GT_revise_230422/'+'GT_' + app + '_revise.csv'
    df_tar = pd.read_csv('GT_revise_230422/'+'GT_' + app + '_revise.csv')
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

    
    #print("get location2:", index, app, search_type)

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
        #print("no sta:", app, id, xpath)

   
    #print("get location3:", index, app, search_type, xpath_type, sta_list)

    #无文件
    map = []
    map_bounds = []
    for path_item in sta_list:
        flag_abs = 0
        try:
            tree = etree.parse('screenshots/'+path_item)
        except:
            #print("get_location failed:", path_item)
            #print(sta_list)
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
        
        #print("get location4:", map)

        #没找到再按绝对位置找
        if not flag_abs:
            
            if (xpath_type == 'absolute') and (search_type != 'id'):
        #        print("get location5:", index, app, xpath, search_type)    
                bounds_item = search_abs(root, xpath)
                if bounds_item:
                    sta = path_item
                    map.append(path_item)
                    map_bounds.append(bounds_item)
                    cnt = cnt + 1

  
    if len(map) == 0:
        print(index, app, id, xpath, sta)
        return sta, bounds, cnt
    
    #uix xml去重 5miles/sign_in.uix
    map_choose = []
    map_bounds_choose = []
    for i in range(0, len(map)):
        p = map[i].split('.')[0]
        if p not in map_choose:
            map_choose.append(p)
            map_bounds_choose.append(map_bounds[i])
    if cnt > 1:
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
                    step += 1
                    if str(y.loc[i]['correct_tgt_index']) != 'nan':
                        dict[y.loc[i]['correct_tgt_index']] = y.loc[i]['index']
                        order[y.loc[i]['correct_tgt_index']] = y.loc[i]['state_name']

            elif flag == 1:
                for key, value in dict.items():
                    if get_same_bool(y.loc[i]['index'], value, y):
                        flag_exist = 1
                if flag_exist == 0:
                    step += 1
                    if str(y.loc[i]['correct_tgt_index']) != 'nan':
                        dict[y.loc[i]['correct_tgt_index']] = y.loc[i]['index']
                        order[y.loc[i]['correct_tgt_index']] = y.loc[i]['state_name']
        else:
            break

    y['step'] = len(dict) 
    y['lcs'] = str(dict)
    y['order'] = str(order)
    return y

def get_same_pos(i, j, x):
    sta_dict_i = {}
    bounds_dict_i = {}
    sta_dict_j = {}
    bounds_dict_j = {}
    flag = 1
    
    if '[' in x.loc[x['index']==i, 'state_name'].values[0]:
        sta_dict_i = ast.literal_eval(x.loc[x['index']==i, 'state_name'].values[0])
        bounds_dict_i = ast.literal_eval(x.loc[x['index']==i, 'bounds'].values[0])
    if '[' in x.loc[x['index']==j, 'state_name'].values[0]:
        sta_dict_j = ast.literal_eval(x.loc[x['index']==j, 'state_name'].values[0])
        bounds_dict_j = ast.literal_eval(x.loc[x['index']==j, 'bounds'].values[0]) 

  
    flag_in = 0
    if len(sta_dict_i) > 0:
        if len(sta_dict_j) > 0:
            for k in range(0, len(sta_dict_i)):
                if sta_dict_i[k] in sta_dict_j:
                    flag_in = 1
                    if bounds_dict_i[k] != bounds_dict_j[k]:
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
        elif str(x.loc[x['index']==i, 'state_name'].values[0]) == 'nan':
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
        #if x.loc[x['index']==i, 'state_name'].values[0]
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

def test(i, j):    
    if data.loc[data['index']==i, 'state_order'].values[0] != data.loc[data['index']==j, 'state_order'].values[0]:
        return -1 if data.loc[data['index']==i, 'state_order'].values[0] < data.loc[data['index']==j, 'state_order'].values[0] else 1
    else:
        if (str(data.loc[data['index']==i, 'tgt_index'].values[0]) == 'nan') or (str(data.loc[data['index']==j, 'tgt_index'].values[0]) == 'nan'):
            if data.loc[data['index']==i, 'src_app'].values[0] == data.loc[data['index']==j, 'src_app'].values[0]:
                if data.loc[i]['tgt_app'] == tgt_app and data.loc[i]['function'] == function:
                    print("tgt=nan src相等 i:", i, data.loc[data['index']==i, 'tgt_index'].values[0], data.loc[data['index']==i, 'src_app'].values[0], data.loc[data['index']==i, 'src_index'].values[0])
                    print("tgt=nan src相等 j:", j, data.loc[data['index']==j, 'tgt_index'].values[0], data.loc[data['index']==j, 'src_app'].values[0], data.loc[data['index']==j, 'src_index'].values[0])

                return -1 if data.loc[data['index']==i, 'src_index'].values[0] < data.loc[data['index']==j, 'src_index'].values[0] else 1
            else:
                if data.loc[i]['tgt_app'] == tgt_app and data.loc[i]['function'] == function:
                    print("tgt=nan src不等 i:", i, data.loc[data['index']==i, 'tgt_index'].values[0], data.loc[data['index']==i, 'src_app'].values[0], data.loc[data['index']==i, 'src_index'].values[0])
                    print("tgt=nan src不等 j:", j, data.loc[data['index']==j, 'tgt_index'].values[0], data.loc[data['index']==j, 'src_app'].values[0], data.loc[data['index']==j, 'src_index'].values[0])

                return -1
        else:
            if data.loc[data['index']==i, 'tgt_index'].values[0] != data.loc[data['index']==j, 'tgt_index'].values[0]:
                return -1 if data.loc[data['index']==i, 'tgt_index'].values[0] < data.loc[data['index']==j, 'tgt_index'].values[0] else 1
            else:
                if data.loc[data['index']==i, 'src_app'].values[0] == data.loc[data['index']==j, 'src_app'].values[0]:
                    return -1 if data.loc[data['index']==i, 'src_index'].values[0] < data.loc[data['index']==j, 'src_index'].values[0] else 1
                else:
                    if data.loc[i]['tgt_app'] == tgt_app and data.loc[i]['function'] == function:
                        print("src不等 i:", i, data.loc[data['index']==i, 'tgt_index'].values[0], data.loc[data['index']==i, 'src_app'].values[0], data.loc[data['index']==i, 'src_index'].values[0])
                        print("src不等 j:", j, data.loc[data['index']==j, 'tgt_index'].values[0], data.loc[data['index']==j, 'src_app'].values[0], data.loc[data['index']==j, 'src_index'].values[0])

                    return -1

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

def fill_pos(x, dict):
    tgt_app = ''
    function = ''
    pmin = 100
    pmax = -100
    cnt_no_state_order = 0
    
    #只有一个src_app 按照现有顺序直接赋值
    if len(x.groupby(['src_app'], group_keys=True).groups) == 1:
        #print(x.iloc[0]['src_app'], x.iloc[0]['tgt_app'], x.iloc[0]['function'])
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
                    

    for i in x.index:
        if (str(x.loc[i]['tgt_index']) != 'nan') or (get_empty_bool(x.loc[i]['index'], x)):
            continue
        if str(x.loc[i]['evaluate_ori']) == 'fn':
            continue
    
        if (x.loc[i]['state_name']) and (x.loc[i]['state_order'] == sys.maxsize):
            cnt_no_state_order = cnt_no_state_order + 1

    
    #有无无法判断的界面顺序
    #(not testSignUp): main/sign_in_or_sign_up/start-> sign_in-> other-> sign_up 
    #(testSignUp):     main/sign_in_or_sign_up/start-> sign_up-> sign_in-> other
    if cnt_no_state_order > 0:
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
                    
    if x.iloc[0]['tgt_app'] == tgt_app and x.iloc[0]['function'] == function:
        print("初始")
        for i in x.index:
            print("index:", x.loc[i]['index'], "tgt:", x.loc[i]['tgt_index'], "state:", x.loc[i]['state_name'], "order:", x.loc[i]['state_order'])

    #x.sort_values(by=['state_order', 'tgt_index', 'by', 'bx', 'src_index', 'index'], inplace=True, ascending=True) 
    x.sort_values(by=['state_order', 'by', 'bx', 'src_index', 'index'], inplace=True, ascending=True) 
      
    
    #x.drop(['bx', 'by', 'bounds'], axis=1, inplace=True)
    
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
                    dict[x.loc[i]['ori_tgt_index']] = x.loc[i]['index']    
                    break
    
    if len(dict) > 0:
        dict_max = max(dict)
    else:
        return x    
    
    
        
    x, dict = single_state_fill(x, dict, dict_max, 1)
    if x.iloc[0]['tgt_app'] == tgt_app and x.iloc[0]['function'] == function:
        print("完成第二步:")
        for i in x.index:
            print("index:", x.loc[i]['index'], "tgt:", x.loc[i]['tgt_index'], "state:", x.loc[i]['state_name'], "order:", x.loc[i]['state_order'])
    
    #多界面尝试
    x.sort_values(by=['state_order', 'src_app', 'tgt_index', 'index'], inplace=True, ascending=True) 
    
    x_line = []
    for i in range(0, len(x)):
        x_line.append(x.iloc[i]['index'])

    for i in range(0, len(x)):
        if (not x.iloc[i]['state_name']) or (x.iloc[i]['state_order'] == sys.maxsize):
            continue     
        if (str(x.iloc[i]['evaluate_ori']) == 'fn') or get_empty_bool(x.iloc[i]['index'], x):
            continue

        if (str(x.iloc[i]['tgt_index']) == 'nan'):
            l = 0
            while x.loc[x['index'] == x_line[l], 'state_order'].values[0] != x.iloc[i]['state_order']:
                l = l + 1

            r = l
            src_list = []
            while r < x_line.index(x.iloc[i]['index']):
                if (x.loc[x['index'] == x_line[r], 'state_order'].values[0] != x.iloc[i]['state_order']):
                    break
                if (x.loc[x['index'] == x_line[r], 'src_app'].values[0] == x.iloc[i]['src_app']) and (x_line[r] != x.iloc[i]['index']):
                    src_list.append(x_line[r])     
                r = r + 1
            
            if (l != i):
                if len(src_list) > 0:
                    if max(src_list) > x.iloc[i]['index']:
                        for j in range(0, len(src_list)):
                            if src_list[j] > x.iloc[i]['index']:
                                p = src_list[j]
                                break
                    else:
                        p_index = x_line.index(src_list[-1]) + 1
                        if x_line[p_index] == x.iloc[i]['index']:
                            p_index = p_index + 1
                        p = x_line[p_index]
                        #print("xxxx:", src_list[-1], x.iloc[i]['index'])
                else:
                    while (r > l):
                        #print(r, x_line[r - 1])
                        if x_line[r - 1] < x.iloc[i]['index']:
                            break
                        else:
                            r = r - 1
                    if x_line[r] == x.iloc[i]['index']:
                        r = r + 1
                    p = x_line[r]

                #print("p:", x_line, p)
                #print("index:", x_line.index(p))
                x_line.remove(x.iloc[i]['index'])
                x_line.insert(x_line.index(p), x.iloc[i]['index'])
                #print("1111111:", x_line)
            
    if x.iloc[0]['tgt_app'] == tgt_app and x.iloc[0]['function'] == function:
        print("多界面排序1:---------------------")
        for i in x.index:
            print("index:", x.loc[i]['index'], "tgt:", x.loc[i]['tgt_index'], "state:", x.loc[i]['state_name'], "order:", x.loc[i]['state_order'])
    
    x = x.reindex(index=x_line)
    #x = x.reindex(index = b)
    if x.iloc[0]['tgt_app'] == tgt_app and x.iloc[0]['function'] == function:
        print("多界面排序2:---------------------")
        for i in x.index:
            print("index:", x.loc[i]['index'], "tgt:", x.loc[i]['tgt_index'], "state:", x.loc[i]['state_name'], "order:", x.loc[i]['state_order'])
    #x.sort_values(by=['state_order', 'src_app', 'tgt_index', 'index'], inplace=True, ascending=True) 
    x, dict = single_state_fill(x, dict, dict_max, 2)
   
    #end
    
    
    if x.iloc[0]['tgt_app'] == tgt_app and x.iloc[0]['function'] == function:
        print("多界面填充后:-----------------")
        for i in x.index:
            print("index:", x.loc[i]['index'], "tgt:", x.loc[i]['tgt_index'], "state:", x.loc[i]['state_name'], "order:", x.loc[i]['state_order'])
    x.sort_values(by=['index'], inplace=True, ascending=True) 
  
    x, dict = no_state_or_order_fill(x, dict)
    
    if x.iloc[0]['tgt_app'] == tgt_app and x.iloc[0]['function'] == function:
        print("全部填充后:-----------------")
        for i in x.index:
            print("index:", x.loc[i]['index'], "tgt:", x.loc[i]['tgt_index'], "state:", x.loc[i]['state_name'], "order:", x.loc[i]['state_order'])
  
    return x


def muti_state(x):
    m = 0
    for i in range(0, len(x)):
        flag = 1
        # if x.iloc[0]['tgt_app'] == 'fivemiles' and x.iloc[0]['function'] == 'testHelp':
        #     print("index:", x.iloc[i]['index'], "cnt: ", x.iloc[i]['cnt'], x.iloc[i]['state_name'])
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
            
            # if (x.iloc[i]['index'] == 553):
            #     print("553", x.iloc[l]['index'], x.iloc[l]['state_name'], x.iloc[r]['index'], x.iloc[r]['state_name'])
            if flagl:
                x.iloc[i, 23] = x.iloc[l]['state_name']
                if (flagr) and (r-i < i-l):
                    x.iloc[i, 23] = x.iloc[r]['state_name']
                x.iloc[i, 24] = bounds_dict[sta_dict.index(x.iloc[i]['state_name'])]
                flag = 1
                m = m +1
            elif flagr:
                x.iloc[i, 23] = x.iloc[r]['state_name']
                x.iloc[i, 24] = bounds_dict[sta_dict.index(x.iloc[i]['state_name'])]
                flag = 1
                m = m +1
        if flag == 1:
            pattern = re.compile(r'\[(.*?)\]')
            locs = pattern.findall(x.iloc[i]['bounds'])
            if (locs):
                x1, y1 = int(locs[0].split(',')[0]), int(locs[0].split(',')[1])
                x2, y2 = int(locs[1].split(',')[0]), int(locs[1].split(',')[1])
                x.iloc[i, 26] = round((x1+x2)/2, 2)
                x.iloc[i, 27] = round((y1+y2)/2, 2)
            x.iloc[i, 25] = 1  #cnt
            x.iloc[i, 28] = str(round(x.iloc[i]['bx'], 2)) + ' ' + str(round(x.iloc[i]['by'], 2))
    
    for i in x.index:
        if x.loc[i]['cnt'] > 1:
            for j in x.index:
                # if x.loc[i]['index'] == 554:
                #     print(x.loc[i]['index'], x.loc[j]['index'], x.loc[j]['state_name'])
                if x.loc[j]['cnt'] == 1:
                    if get_same_bool(x.loc[i]['index'], x.loc[j]['index'], x):
                    # if x.loc[i]['index'] == 554:
                    #     print("same", x.loc[i]['index'], x.loc[j]['index'], x.loc[j]['state_name'])
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
    #print(x)
    #tgt_index22 state_name23 bounds24 cnt25 bx26 by27 pos28
    for i in x.index:
        x.loc[i, 'state_name'], x.loc[i, 'bounds'], x.loc[i, 'cnt'] = get_location(x.loc[i]['index'], x.loc[i]['tgt_app'], str(x.loc[i]['tgt_add_id']), str(x.loc[i]['tgt_add_xpath']))
        x.loc[i, 'bx'] = sys.maxsize
        x.loc[i, 'by'] = sys.maxsize
        x.loc[i, 'pos'] = str(sys.maxsize) + ' ' + str(sys.maxsize)
        x.loc[i, 'list'] = x.loc[i, 'state_name']
    
    
    x.sort_values(by=['src_app', 'index'], inplace=True, ascending=True) 

    # if x.iloc[0]['tgt_app'] == 'fivemiles' and x.iloc[0]['function'] == 'testSearch':
    #     for i in x.index:
    #         print("index:", x.loc[i]['index'], "src_app", x.loc[i]['src_app'], "state:", x.loc[i]['state_name'])
    

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
        # if x.iloc[0]['tgt_app'] == 'fivemiles' and x.iloc[0]['function'] == 'testSearch':
        #     print(lcs_dict)
        #按照最长子序列填充tgt_index + 记录界面相对位置
        for i in x.index:
            #按correctindex对应的组件填写x.loc[i, 'tgt_index']
            for key, value in lcs_dict.items():
                if (str(x.loc[i]['evaluate_ori']) == 'fn') or get_empty_bool(x.loc[i]['index'], x):
                    continue
                if get_same_bool(x.loc[i]['index'], value, x):
                    x.loc[i, 'tgt_index'] = key
                    break
            for key, value in order_dict.items():
                if (str(x.loc[i]['evaluate_ori']) == 'fn') or get_empty_bool(x.loc[i]['index'], x):
                    continue
                if value == x.loc[i]['state_name']:
                    x.loc[i, 'state_order'] = key
                    break
        # if x.iloc[0]['tgt_app'] == 'home' and x.iloc[0]['function'] == 'testAccount':
        #     p = x
        #     p.to_csv("check.csv", index=False)
  
    
    # else:
    #     #print("failed:", x.iloc[0]['tgt_app'], x.iloc[0]['function'])
    #     df_lcs = pd.read_csv('correct_lcs.csv')
    #     result_group_src = df_lcs.groupby(['tgt_app', 'function']).get_group((x.iloc[0]['tgt_app'], x.iloc[0]['function']))
    #     flag = 0
    #     for i in result_group_src.index:
    #         if str(result_group_src.loc[i]['tgt_index']) != 'nan':
    #             flag = 1
    #             lcs_dict[result_group_src.loc[i]['tgt_index']] = result_group_src.loc[i]['index']
    #             order_dict[result_group_src.loc[i]['tgt_index']] = result_group_src.loc[i]['state_name']
              

    #x.sort_values(by=['state_order', 'by', 'bx', 'tgt_index', 'index'], inplace=True, ascending=True) 
    #x.drop(['bx', 'by', 'bounds'], axis=1, inplace=True)
    
               
    # if (x.iloc[0]['tgt_app'] == 'a31' and x.iloc[0]['function'] == 'b31'):
    #     print(x)
    #     print(lcs_idx, len(lcs_dict), lcs_dict)
    #     print(order_dict)
    
    #2.按照相对位置填充tgt_index
    x = fill_pos(x, lcs_dict)

    #检查
    x.sort_values(by=['tgt_index'], inplace=True, ascending=True) 
    l = 0
    global problem, pro_cnt
    for i in range(1, len(x)):
        if x.iloc[i]['tgt_index_rev'] < x.iloc[l]['tgt_index_rev']:
           # problem.loc[pro_cnt] = ['index', x.iloc[i]['index']]
            if get_empty_bool(x.iloc[i]['index'], x) and (str(x.iloc[i]['tgt_index']) == 'nan'):
                problem.append([x.iloc[i]['index'], x.iloc[i]['tgt_app'], x.iloc[i]['function'], x.iloc[i]['tgt_index'], x.iloc[i]['tgt_index_rev'], 'all empty'])
            else:
                problem.append([x.iloc[i]['index'], x.iloc[i]['tgt_app'], x.iloc[i]['function'], x.iloc[i]['tgt_index'], x.iloc[i]['tgt_index_rev'], ''])

        elif x.iloc[i]['tgt_index_rev'] == x.iloc[l]['tgt_index_rev']:
            if x.iloc[i]['tgt_index'] == x.iloc[l]['tgt_index']:
                l = i
            else:
                if get_empty_bool(x.iloc[i]['index'], x) and (str(x.iloc[i]['tgt_index']) == 'nan'):
                    problem.append([x.iloc[i]['index'], x.iloc[i]['tgt_app'], x.iloc[i]['function'], x.iloc[i]['tgt_index'], x.iloc[i]['tgt_index_rev'], 'all empty'])
                else:
                    problem.append([x.iloc[i]['index'], x.iloc[i]['tgt_app'], x.iloc[i]['function'], x.iloc[i]['tgt_index'], x.iloc[i]['tgt_index_rev'], ''])
        else:
            if x.iloc[i]['tgt_index'] == x.iloc[l]['tgt_index']:
                if get_empty_bool(x.iloc[i]['index'], x) and (str(x.iloc[i]['tgt_index']) == 'nan'):
                    problem.append([x.iloc[i]['index'], x.iloc[i]['tgt_app'], x.iloc[i]['function'], x.iloc[i]['tgt_index'], x.iloc[i]['tgt_index_rev'], 'all empty'])
                else:
                    problem.append([x.iloc[i]['index'], x.iloc[i]['tgt_app'], x.iloc[i]['function'], x.iloc[i]['tgt_index'], x.iloc[i]['tgt_index_rev'], ''])
            else:
                l = i
    return x

def main():
    df = pd.read_csv('data_new.csv')
    #df = pd.read_csv('part.csv')
    #组内找相对位置+最长子序列
    result_group = df.groupby(['tgt_app', 'function'], group_keys=True).apply(solve)
    #反向填充原始文件
    for i in range(0, len(result_group)):
        index = result_group.iloc[i]['index']
        df.loc[df['index']==index, 'tgt_index'] = result_group.iloc[i]['tgt_index']
        df.loc[df['index']==index, 'state_name'] = result_group.iloc[i]['state_name']
        df.loc[df['index']==index, 'pos'] = result_group.iloc[i]['pos']
        df.loc[df['index']==index, 'list'] = result_group.iloc[i]['list']
    
    problem_df = pd.DataFrame(problem)
    problem_df.rename(columns={0: 'index'}, inplace=True)
    problem_df.rename(columns={1: 'tgt_app'}, inplace=True)
    problem_df.rename(columns={2: 'function'}, inplace=True)
    problem_df.rename(columns={3: 'tgt_index'}, inplace=True)
    problem_df.rename(columns={4: 'tgt_index_rev'}, inplace=True)
    problem_df.rename(columns={5: 'reason'}, inplace=True)

    problem_df.to_csv("problem_dataset2_0530.csv", index=False)
    df.to_csv("result_dataset2_0530.csv", index=False)
    

if __name__ == "__main__":
    main()
