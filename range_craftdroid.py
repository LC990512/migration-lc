import pandas as pd
import numpy as np
import sys
import math
import re
import ast
import os
from lxml import etree

num = 0
pro_cnt = 0
data = pd.DataFrame()
problem = []

#tgtindex=20列
#state_name=21列
def search_xml(node, search_type, id, re_xpath):
    if search_type == 'id':
        if node.get('resource-id') == id:
            return(node.get('bounds'))
    elif search_type == 'xpath':
        if node.get('class') == re_xpath['class'] and node.get('content-desc') == re_xpath['cd']:
            return(node.get('bounds'))
    if len(node) > 0:
        for child in node:
            bounds = search_xml(child, search_type, id, re_xpath)
            if bounds: return bounds
    return None

def get_location(app, id, xpath):
    sta = ''
    bounds = ''
    cnt = 0
    
    if (str(id) == 'nan') and (str(xpath) == 'nan'):
    #    print("none id,xpath:", app, id, xpath)
        return sta, bounds, cnt
    
    
    #先从revise.csv中找sta
    df_tar = pd.read_csv('GT_revise_0711_dumplicate/'+app[:2]+'/'+app+'_revise.csv')
    sta_list = []
    re_xpath = {'class':"", 'cd':""}    
    if (str(xpath) != 'nan'):
        pattern = re.compile(r'\/\/(.*?)\[')
        if (pattern.findall(xpath)):
            re_xpath['class'] = pattern.findall(xpath)[0]
        pattern = re.compile(r'\"(.*?)\"')
        if (pattern.findall(xpath)):
            re_xpath['cd'] = pattern.findall(xpath)[0]
    
    for i in df_tar.index:
        #ori_id // ori_appium_xpath
        if id == df_tar.loc[i]['ori_id']:
            if (xpath == 'nan'):
                search_type = 'id'
                sta = df_tar.loc[i]['state_name']
                sta_list.append(df_tar.loc[i]['state_name'])
            if  (xpath == df_tar.loc[i]['ori_appium_xpath']):
                search_type = 'both'
                sta = df_tar.loc[i]['state_name']
                sta_list.append(df_tar.loc[i]['state_name'])
        elif id == 'nan':
            if xpath == df_tar.loc[i]['ori_appium_xpath']:
                search_type = 'xpath'
                sta = df_tar.loc[i]['state_name']
                sta_list.append(df_tar.loc[i]['state_name'])

    #/Users/lccc/Desktop/pku/android test /screenshots/a1/a12
    
    
    #遍历不同的xml处理无sta的情况
    flag = 0
    mydir = 'screenshots/'+app[:2]+'/'+app
    if not sta:
        if (str(xpath) == 'nan'):
            search_type = 'id'
        elif (str(id) == 'nan'):
            search_type = 'xpath'
        else:
            search_type = 'both'
        
        for root, dirs, files in os.walk(mydir):
            for file in files:
                if not file.endswith(".xml"):
                    continue
                targetFilePath = app+'/'+ file
                sta_list.append(targetFilePath)
                flag = 1
    #    print("no sta:", app, id, xpath)
    #    return sys.maxsize, sys.maxsize, sta
    
    
    #无文件
    map = []
    map_bounds = []
    for path_item in sta_list:
        try:
            tree = etree.parse('screenshots/'+app[:2]+'/'+app+'/'+path_item[4:])
        except:
            #print("get_location failed:", 'screenshots/'+app[:2]+'/'+app+'/'+path_item[4:])
            i = sta_list.index(path_item)
            continue
            #return sta, bounds, cnt
        root = tree.getroot()
        for node in root:
            if search_type == 'both':
                bounds_item = search_xml(node, 'id', id, re_xpath)
                if not bounds_item:
                    bounds_item = search_xml(node, 'xpath', id, re_xpath)
            else:
                bounds_item = search_xml(node, search_type, id, re_xpath)
            if bounds_item:
                bounds = bounds_item
                sta = path_item
                map.append(path_item)
                map_bounds.append(bounds_item)
                cnt = cnt + 1

    #none sta: a23 org.secuso.privacyfriendlytodolist:id/et_todo_list_name nan  6
    
    if not bounds:    
        # if not sta:
        #     print("none sta:", app, id, xpath, sta, len(sta_list))
        # else:
        #     print("none bounds:", app, id, xpath, sta, len(sta_list))
        return sta, bounds, cnt
    
    
    if cnt > 1:
#        print("多个sta界面:", app, id, xpath, sta, cnt)
#        print(map_bounds)
        sta = str(map)
        bounds = str(map_bounds)

    return sta, bounds, cnt


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
        if (y.loc[i]['ori_predict_label'] == 1) and (y.loc[i]['label'] == 1) or (str(y.loc[i]['type']) == 'SYS_EVENT'):
            flag_exist = 0
            if y.loc[i]['src_index'] == first_src_index:
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

#是否为空组件
def get_empty_bool(i, x):
    #Tgt_add_id and tgt_add_xpath and tgt_text and tgt_content 都为空的时候不给tgt_index 
    if (str(x.loc[x['index']==i, 'tgt_add_id'].values[0]) == 'nan') and (str(x.loc[x['index']==i, 'tgt_add_xpath'].values[0]) == 'nan') and (str(x.loc[x['index']==i, 'tgt_text'].values[0]) == 'nan') and (str(x.loc[x['index']==i, 'tgt_content'].values[0]) == 'nan'):
        return True
    return False

#pos是否每个界面内都一致
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
    if len(sta_dict_i) > 0:
        if len(sta_dict_j) > 0:
            for k in range(0, len(sta_dict_i)):
                if sta_dict_i[k] in sta_dict_j:
                    if bounds_dict_i[k] != bounds_dict_j[k]:
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

#是否为同一组件
def get_same_bool(i, j, x):
    #predict_input canonical
    special = 'clear_and_send_keys_and_hide_keyboard'
    stri = str(x.loc[x['index']==i, 'tgt_text'].values[0])
    strj = str(x.loc[x['index']==j, 'tgt_text'].values[0])
                
    if (str(x.loc[x['index']==i, 'tgt_text'].values[0]) != str(x.loc[x['index']==j, 'tgt_text'].values[0])):
        if (j == 570):
            print(i, j, stri, strj)  
        flag = 0
        if str(x.loc[x['index']==i, 'tgt_text'].values[0]) == '0' and str(x.loc[x['index']==j, 'tgt_text'].values[0]) == '65.09':
            flag = 1
        if str(x.loc[x['index']==i, 'tgt_text'].values[0]) == '65.09' and str(x.loc[x['index']==j, 'tgt_text'].values[0]) == '0':
            flag = 1
        if str(x.loc[x['index']==i, 'tgt_text'].values[0]).startswith("https"):
            stri = stri[0:stri.rfind('com')]
            strj = strj[0:strj.rfind('com')]      
            if (stri == strj):
                flag = 1
        if (flag == 0):
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

def set_order(x, flag):
    if flag == 1:
        for i in x.index:
            if (not x.loc[i]['state_name']):
                continue
            name = x.loc[i]['state_name'].split('/')[1]
            
            if name == 'start.xml':
                x.loc[i, 'state_order'] = 1
            elif name == 'signup.xml':
                x.loc[i, 'state_order'] = 2
            elif name == 'signin.xml':
                x.loc[i, 'state_order'] = 3
            else:
                x.loc[i, 'state_order'] = 4
    else:
        for i in x.index:
            if (not x.loc[i]['state_name']):
                continue
            name = x.loc[i]['state_name'].split('/')[1]

            if name == 'start.xml':
                x.loc[i, 'state_order'] = 1
            elif name == 'signin.xml':
                x.loc[i, 'state_order'] = 2
            elif name == 'signup.xml':
                x.loc[i, 'state_order'] = 3
            else:
                x.loc[i, 'state_order'] = 4
    return x

def fill_pos(x, dict, dict_max):
    pmin = 100
    pmax = -100
    tgt = 'a32'
    function = 'b31'

    #对于b31: 界面顺序 start->signup->signin->others
    #对于b32: 界面顺序 start->signin->signup->others

    #a31/a34 b31 start->signup->signin->others
    #a32 b31 signin signup
    #a34 b32 start->signin->signup->others
    if x.iloc[0]['function'] == 'b31':
        if x.iloc[0]['tgt_app'] == 'b32':
            x = set_order(x, 2)
        else:
            x = set_order(x, 1)
        

    if x.iloc[0]['function'] == 'b32':
        x = set_order(x, 2)
    #
    # if x.iloc[0]['function'] == 'b31':
    #     for i in x.index:
    #         if (not x.loc[i]['state_name']):
    #             continue
    #         name = x.loc[i]['state_name'].split('/')[1]
            
    #         if name == 'start.xml':
    #             x.loc[i, 'state_order'] = 1
    #         elif name == 'signup.xml':
    #             x.loc[i, 'state_order'] = 2
    #         elif name == 'signin.xml':
    #             x.loc[i, 'state_order'] = 3
    #         else:
    #             x.loc[i, 'state_order'] = 4
    # if x.iloc[0]['function'] == 'b32':
    #     for i in x.index:
    #         if (not x.loc[i]['state_name']):
    #             continue
    #         name = x.loc[i]['state_name'].split('/')[1]

    #         if name == 'start.xml':
    #             x.loc[i, 'state_order'] = 1
    #         elif name == 'signin.xml':
    #             x.loc[i, 'state_order'] = 2
    #         elif name == 'signup.xml':
    #             x.loc[i, 'state_order'] = 3
    #         else:
    #             x.loc[i, 'state_order'] = 4

 
        
   
    x.sort_values(by=['state_order', 'by', 'bx', 'tgt_index', 'src_app', 'index'], inplace=True, ascending=True) 
    
    if (x.iloc[0]['tgt_app'] == tgt and x.iloc[0]['function'] == function):
        #print(x)
        x.to_csv("part.csv", index=False)
    

    for i in range(len(x)-1, -1, -1):
        #sys,empty,找不到sta, 填完的跳过
        if ((str(x.iloc[i]['type']) == 'EMPTY_EVENT') or (str(x.iloc[i]['type']) == 'SYS_EVENT')) :
            continue
        if (not x.iloc[i]['state_name']):
            continue
        if str(x.iloc[i]['tgt_index']) != 'nan':
            continue
        if x.iloc[i]['state_order'] == sys.maxsize:
            continue
    
        
        #判断是否已有该组件
        flag = 0
        for key, value in dict.items():
            if get_same_bool(x.iloc[i]['index'], value, x):
                x.iloc[i, 20] = key
                flag = 1
        if (flag == 1):
            continue
        
        #寻找左右区间，第一个有值且在同一界面内的
        l = i  
        r = i
        #while ((str(x.iloc[l]['tgt_index']) == 'nan') or not get_screen_bool(i, l, x)):
        while ((str(x.iloc[l]['tgt_index']) == 'nan')):
            if (l - 1 < 0):
                break
            l = l - 1
        while ((str(x.iloc[r]['tgt_index']) == 'nan')):
            if (r + 1 == len(x)):
                break
            r = r + 1
        # if (x.iloc[0]['tgt_app'] == tgt and x.iloc[0]['function'] == function):
        #     print("左右区间l, r:", x.iloc[i]['index'], x.iloc[l]['index'], x.iloc[r]['index'])
        
    
        # if (x.iloc[0]['tgt_app'] == 'a12' and x.iloc[0]['function'] == 'b12'):
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
                x.iloc[i, 20] = index
                dict[index] = x.iloc[i]['index']    
        else:
            if get_screen_bool(i, l, x) or not get_screen_bool(i, l, x):
                #左右都有
                if ((str(x.iloc[r]['tgt_index']) != 'nan')):

                    # if (x.iloc[0]['tgt_app'] == 'a12' and x.iloc[0]['function'] == 'b12'):
                    #     print("左右都有l, r:", x.iloc[i]['index'], x.iloc[l]['index'], x.iloc[r]['index'])
        
    
                    #左右相等 看一下=5的例子
                    if get_same_bool(x.iloc[l]['index'], x.iloc[r]['index'], x):
                        index = x.iloc[r]['tgt_index'] - 0.1
                        while ((index in dict.keys()) or ((index > -3) and (index < 0))):
                            index = index - 0.1
                    else:    
                        # if (x.iloc[0]['tgt_app'] == 'a12' and x.iloc[0]['function'] == 'b12'):
                        #     print("左右不等l, r:", x.iloc[i]['index'], x.iloc[l]['index'], x.iloc[r]['index'])
        
                
                        index = (x.iloc[l]['tgt_index'] + x.iloc[r]['tgt_index']) / 2   
                        while (index in dict.keys()):
                            index = (x.iloc[l]['tgt_index'] + index) / 2
                    x.iloc[i, 20] = index
                    dict[index] = x.iloc[i]['index']    
                #只有左侧
                else:
                    index = x.iloc[l]['tgt_index'] + 1
                    while ((index in dict.keys()) or ((index > dict_max) and (index < 20))):
                        if index in dict.keys():
                            if get_same_bool(x.iloc[i]['index'], dict[index], x):
                                break
                        index = index + 1
                    x.iloc[i, 20] = index
                    dict[index] = x.iloc[i]['index']
                            
           
            #else什么都没有
        pmin = min(pmin, x.iloc[i]['tgt_index'])
        pmax = max(pmax, x.iloc[i]['tgt_index'])

    for i in range(0, len(x)):
        if str(x.iloc[i]['tgt_index']) != 'nan':
            continue
        if ((str(x.iloc[i]['type']) == 'EMPTY_EVENT') or (str(x.iloc[i]['type']) == 'SYS_EVENT')) :
            continue
        if (not x.iloc[i]['state_name'] or x.iloc[i]['state_order'] == sys.maxsize):
            flag = 0
            if not x.iloc[i]['state_name']:
                y = x[x['src_app'] == x.iloc[i]['src_app']]
                for j in y.index:
                    if (y.loc[j]['tgt_index'] == 0) and (x.iloc[i]['src_index'] < y.loc[j]['src_index']):
                            x.iloc[i, 20] = -3
                            flag = 1
                            break
            if flag == 0:
                for key, value in dict.items():
                    # if x.iloc[i]['index'] == 1244:
                    #     print("get_same_bool:", x.iloc[i]['index'], value)
                    if get_same_bool(x.iloc[i]['index'], value, x):
                        x.iloc[i, 20] = key
                        flag = 1
                        break
            
            if flag == 0:
                # y = x[x['src_app'] == x.iloc[i]['src_app']]
                # y.sort_values(by=['index'], inplace=True, ascending=True) 
                index =  max(max(dict) + 1, 20)
                x.iloc[i, 20] = index
                dict[index] = x.iloc[i]['index']      
    
    return x
def muti_state(x):
    sta_dict = {}
    bounds_dict = {}
    m = 0
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
                x.iloc[i, 22] = x.iloc[l]['state_name']
                if (flagr) and (r-i < i-l):
                    x.iloc[i, 22] = x.iloc[r]['state_name']
                x.iloc[i, 23] = bounds_dict[sta_dict.index(x.iloc[i]['state_name'])]
                flag = 1
                m = m +1

            elif flagr:
                x.iloc[i, 22] = x.iloc[r]['state_name']
                x.iloc[i, 23] = bounds_dict[sta_dict.index(x.iloc[i]['state_name'])]
                flag = 1
                m = m +1

        if flag == 1:
            pattern = re.compile(r'\[(.*?)\]')
            locs = pattern.findall(x.iloc[i]['bounds'])
            if (locs):
                x1, y1 = int(locs[0].split(',')[0]), int(locs[0].split(',')[1])
                x2, y2 = int(locs[1].split(',')[0]), int(locs[1].split(',')[1])
                x.iloc[i, 25] = round((x1+x2)/2, 2)
                x.iloc[i, 26] = round((y1+y2)/2, 2)
            x.iloc[i, 24] = 1  #cnt
            x.iloc[i, 27] = str(round(x.iloc[i]['bx'], 2)) + ' ' + str(round(x.iloc[i]['by'], 2))
    
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
    tgt = 'a12'
    function = 'b11'

    global num
    num = num + 1
    #state_name22列 bounds23 cnt24 bx25 by26 pos27 list28 same29
    for i in x.index:
        x.loc[i, 'state_name'], x.loc[i, 'bounds'], x.loc[i, 'cnt'] = get_location(x.loc[i]['tgt_app'], str(x.loc[i]['tgt_add_id']), str(x.loc[i]['tgt_add_xpath']))
        x.loc[i, 'bx'] = sys.maxsize
        x.loc[i, 'by'] = sys.maxsize
        x.loc[i, 'pos'] = str(sys.maxsize) + ' ' + str(sys.maxsize)
        x.loc[i, 'list'] = x.loc[i, 'state_name']
        x.loc[i, 'same'] = 0
    # if (x.iloc[0]['tgt_app'] == 'a12' and x.iloc[0]['function'] == 'b12'):
    #     print(x)
    
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
    if (lcs_max > 0):
        lcs_dict = ast.literal_eval(result_group_src.iloc[lcs_idx]['lcs'])
        order_dict = ast.literal_eval(result_group_src.iloc[lcs_idx]['order'])
    else:
        df_lcs = pd.read_csv('correct_lcs.csv')
        result_group_src = df_lcs.groupby(['tgt_app', 'function'], group_keys=True).get_group((x.iloc[0]['tgt_app'], x.iloc[0]['function']))
        for i in result_group_src.index:
            if str(result_group_src.loc[i]['tgt_index']) != 'nan':
                lcs_dict[result_group_src.loc[i]['tgt_index']] = result_group_src.loc[i]['index']
                order_dict[result_group_src.loc[i]['tgt_index']] = result_group_src.loc[i]['state_name']
    

  
    #按照最长子序列填充tgt_index + 记录界面相对位置
    for i in x.index:
        x.loc[i, 'state_order'] = sys.maxsize
        #sys_event直接填
        if str(x.loc[i]['type']) == 'SYS_EVENT':
            x.loc[i, 'tgt_index'] = x.loc[i]['correct_tgt_index']
            lcs_dict[x.loc[i]['correct_tgt_index']] = x.loc[i]['index']
            continue
        #按correctindex对应的组件填写x.loc[i, 'tgt_index']
        for key, value in lcs_dict.items():
            if get_same_bool(x.loc[i]['index'], value, x):
                x.loc[i, 'tgt_index'] = key
                break
        for key, value in order_dict.items():
            if value == x.loc[i]['state_name']:
                x.loc[i, 'state_order'] = key
                break
            

    
    # x.sort_values(by=['state_order', 'by', 'bx', 'tgt_index', 'index'], inplace=True, ascending=True) 
    # x.drop(['bx', 'by', 'bounds'], axis=1, inplace=True)
    
               
    # if (x.iloc[0]['tgt_app'] == tgt and x.iloc[0]['function'] == function):
    #     print(x)
    #     print(lcs_idx, len(lcs_dict), lcs_dict)
    #     print(order_dict)
    
    #2.按照相对位置填充tgt_index
    #if len(lcs_dict) > 0:
    x = fill_pos(x, lcs_dict, max(lcs_dict))

    # if (x.iloc[0]['tgt_app'] == 'a31' and x.iloc[0]['function'] == 'b31'):
    #     print(x)
    
    #检查
    x.sort_values(by=['tgt_index', 'index'], inplace=True, ascending=True) 
    l = 0
    appear_index = []
    appear_tgt_rev = []
    appear_index.append(x.iloc[0]['index'])
    appear_tgt_rev.append(x.iloc[0]['tgt_index_rev'])
    
    global problem, pro_cnt
    for i in range(1, len(x)):
        if (x.iloc[i]['index'] == 226):
            print(x.iloc[i]['index'], x.iloc[l]['index']) 
        if x.iloc[i]['tgt_index_rev'] < x.iloc[l]['tgt_index_rev']:
           # problem.loc[pro_cnt] = ['index', x.iloc[i]['index']]
            if (not x.iloc[i]['state_name']) and (x.iloc[i]['tgt_index_rev'] == -3):
                problem.append([x.iloc[i]['index'], x.iloc[i]['tgt_app'], x.iloc[i]['function'], x.iloc[i]['tgt_index'], x.iloc[i]['tgt_index_rev'], 'cannot find state'])
            elif get_same_bool(x.iloc[i]['index'], x.iloc[l]['index'], x):
                x.iloc[i, 29] = 'same object'
                problem.append([x.iloc[i]['index'], x.iloc[i]['tgt_app'], x.iloc[i]['function'], x.iloc[i]['tgt_index'], x.iloc[i]['tgt_index_rev'], 'same object'])
            elif x.iloc[i]['tgt_index_rev'] in appear_tgt_rev:
                    if not get_same_bool(x.iloc[i]['index'], appear_index[appear_tgt_rev.index(x.iloc[i]['tgt_index_rev'])], x):
                        problem.append([x.iloc[i]['index'], x.iloc[i]['tgt_app'], x.iloc[i]['function'], x.iloc[i]['tgt_index'], x.iloc[i]['tgt_index_rev'], 'not same object'])
        
            else:
                problem.append([x.iloc[i]['index'], x.iloc[i]['tgt_app'], x.iloc[i]['function'], x.iloc[i]['tgt_index'], x.iloc[i]['tgt_index_rev'], ''])

        elif x.iloc[i]['tgt_index_rev'] == x.iloc[l]['tgt_index_rev']:
            if x.iloc[i]['tgt_index'] == x.iloc[l]['tgt_index']:
                l = i
                appear_index.append(x.iloc[l]['index'])
                appear_tgt_rev.append(x.iloc[l]['tgt_index_rev'])
            else:
                if (not x.iloc[i]['state_name']) and (x.iloc[i]['tgt_index_rev'] == -3):
                    problem.append([x.iloc[i]['index'], x.iloc[i]['tgt_app'], x.iloc[i]['function'], x.iloc[i]['tgt_index'], x.iloc[i]['tgt_index_rev'], 'cannot find state'])
                elif get_same_bool(x.iloc[i]['index'], x.iloc[l]['index'], x):
                    x.iloc[i, 29] = 'same object'
                    problem.append([x.iloc[i]['index'], x.iloc[i]['tgt_app'], x.iloc[i]['function'], x.iloc[i]['tgt_index'], x.iloc[i]['tgt_index_rev'], 'same object'])
                elif x.iloc[i]['tgt_index_rev'] in appear_tgt_rev:
                    if not get_same_bool(x.iloc[i]['index'], appear_index[appear_tgt_rev.index(x.iloc[i]['tgt_index_rev'])], x):
                        problem.append([x.iloc[i]['index'], x.iloc[i]['tgt_app'], x.iloc[i]['function'], x.iloc[i]['tgt_index'], x.iloc[i]['tgt_index_rev'], 'not same object'])
        
                else:
                    problem.append([x.iloc[i]['index'], x.iloc[i]['tgt_app'], x.iloc[i]['function'], x.iloc[i]['tgt_index'], x.iloc[i]['tgt_index_rev'], ''])
        else:
            if x.iloc[i]['tgt_index'] == x.iloc[l]['tgt_index']:
                if (not x.iloc[i]['state_name']) and (x.iloc[i]['tgt_index_rev'] == -3):
                    problem.append([x.iloc[i]['index'], x.iloc[i]['tgt_app'], x.iloc[i]['function'], x.iloc[i]['tgt_index'], x.iloc[i]['tgt_index_rev'], 'cannot find state'])
                elif get_same_bool(x.iloc[i]['index'], x.iloc[l]['index'], x):
                    x.iloc[i, 29] = 'same object'
                    problem.append([x.iloc[i]['index'], x.iloc[i]['tgt_app'], x.iloc[i]['function'], x.iloc[i]['tgt_index'], x.iloc[i]['tgt_index_rev'], 'same object'])
                elif x.iloc[i]['tgt_index_rev'] in appear_tgt_rev:
                    if not get_same_bool(x.iloc[i]['index'], appear_index[appear_tgt_rev.index(x.iloc[i]['tgt_index_rev'])], x):
                        problem.append([x.iloc[i]['index'], x.iloc[i]['tgt_app'], x.iloc[i]['function'], x.iloc[i]['tgt_index'], x.iloc[i]['tgt_index_rev'], 'not same object'])
        
                else:
                    problem.append([x.iloc[i]['index'], x.iloc[i]['tgt_app'], x.iloc[i]['function'], x.iloc[i]['tgt_index'], x.iloc[i]['tgt_index_rev'], ''])
            else:
                l = i
                appear_index.append(x.iloc[l]['index'])
                appear_tgt_rev.append(x.iloc[l]['tgt_index_rev'])
            

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
        df.loc[df['index']==index, 'same'] = result_group.iloc[i]['same']
    
    problem_df = pd.DataFrame(problem)
    problem_df.rename(columns={0: 'index'}, inplace=True)
    problem_df.rename(columns={1: 'tgt_app'}, inplace=True)
    problem_df.rename(columns={2: 'function'}, inplace=True)
    problem_df.rename(columns={3: 'tgt_index'}, inplace=True)
    problem_df.rename(columns={4: 'tgt_index_rev'}, inplace=True)
    problem_df.rename(columns={5: 'reason'}, inplace=True)

    problem_df.to_csv("problem_craftdroid_0609.csv", index=False)
    df.to_csv("result_craftdroid_0609.csv", index=False)
    

if __name__ == "__main__":
    main()
