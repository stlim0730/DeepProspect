from __future__ import division
import sys
import time
import string
import requests
from bs4 import BeautifulSoup
import pandas
import fpformat

STD_P_FIELD_CNT = 35
VAL_P_FIELD_CNT = 24
STD_H_FIELD_CNT = 30
VAL_H_FIELD_CNT = 24

POSITION_MAP = {'1': 'P', '2': 'C', '3':'1B', '4':'2B', '5':'3B', '6': 'SS', '7': 'LF', '8': 'CF', '9': 'RF', 'D': 'DH', 'O': 'OF'}

root_url = 'http://www.baseball-reference.com'
player_index_url = 'http://www.baseball-reference.com/players/'
pitcher_cnt = 0
pitcher_rec_cnt = 0
hitter_cnt = 0
hitter_rec_cnt = 0
player_cnt = 0 #18,450 expected in total

p_header = ['PID', 'NAME', 'AGE', 'T', 'YR', 'TM', 'G', 'GS', 'W', 'L', 'WLP', 'CG', 'ShO', 'IP', 'H', 'R', 'ER', 'TBB', 'IBB', 'SO', 'ERA', 'GF', 'SV', 'HR', 'HB', 'BK', 'WP', 'SO/BB', 'H/9', 'SO/9', 'BB/9', 'HR/9', 'BR/9', 'RAA', 'WAA', 'WAAadj', 'WAR', 'RAR', 'waaWL%', '162WL%']
h_header = ['PID', 'NAME', 'AGE', 'B', 'YR', 'TM', 'POS', 'G', 'PA', 'AB', 'R', 'H', '2B', '3B', 'HR', 'RBI', 'RP', 'TBB', 'IBB', 'SO', 'SB', 'CS', 'SBP', 'AVG', 'OBP', 'SLG', 'OPS', 'SH', 'SF', 'HBP', 'TB', 'GDP', 'OPS+', 'RCbasic', 'RCsb', 'RCtech', 'RAA', 'WAA', 'RAR', 'WAR', 'waaWL%', '162WL%', 'oWAR', 'dWAR', 'oRAR']
# Missing fields compared to the agency dataset:
# Pitching: DAYS, MLS, QS, QSP, SvOp, SvPct, BlSv, IR, IRS, IRSP, Hld, DL, 2B, 3B, OBA, OOBP, OSLG
# Hitting: DAYS, MLS, GS, DL

p_data = []
h_data = []
p_file = 'PitcherData_BR_'
h_file = 'HitterData_BR_'
target_initial = 'a'

def getTimeString():
  return time.strftime("%H:%M:%S")

def timedLog(str):
  print "#scrapper [" + getTimeString() + "] - " + str

def get_data_from_cols(columns, col_index, floating_point=None):
  raw = columns[col_index]
  val = None
  if col_index == 0:
    val = str(raw.contents[0]).encode('utf-8') # To avoid allstar mark
  else:
    val = str(raw.string).encode('utf-8')
  if val == 'None':
    return None
  if floating_point:
    val = fpformat.fix(float(val), floating_point)
    return str(val)
  else:
    return val

def get_position(num_str):
  if not num_str:
    return None
  val = str(num_str)
  for c in val:
    if c in POSITION_MAP:
      return POSITION_MAP[c]
  return None

# Command line args
if len(sys.argv) == 2:
  target_initial = sys.argv[-1]

for lastname_initial in [target_initial]:#string.ascii_lowercase:
  # Players with last names starting with lastname_initial
  timedLog('reading baseball players\' data whose last names start with ' + lastname_initial)
  letter_page_res = requests.get(player_index_url + lastname_initial)
  letter_page = letter_page_res.text
  letter_page_obj = BeautifulSoup(letter_page)
  blockquotes = letter_page_obj.find_all('blockquote')
  for blockquote in blockquotes:
    player_links = blockquote.find_all('a')
    for player_link in player_links:
      player_url = root_url + player_link['href']
      player_page_res = requests.get(player_url)
      player_page = player_page_res.text
      player_page_obj = BeautifulSoup(player_page)
      player_name = player_page_obj.find('span', id='player_name').string.encode('utf-8')
      positions_obj = player_page_obj.select('span[itemprop="role"]')
      if len(positions_obj) >=1:
        player_positions_obj = positions_obj[0]
      else:
        print player_name + ' has wrong position page.'
        continue
      player_positions = player_positions_obj.string.encode('utf-8').split(',')
      if 'Pitcher' in player_positions:
        pitcher_cnt += 1
        arm = player_positions_obj.next_sibling.next_sibling.next_sibling.next_sibling.next_sibling.next_element.next_element.strip()
        player_std_p_table = player_page_obj.find('table', id='pitching_standard').tbody.find_all('tr')
        player_val_p_table = player_page_obj.find('table', id='pitching_value').tbody.find_all('tr')
        for std_row in player_std_p_table:
          if 'blank_table' in std_row['class']:
            continue
          std_cols = std_row.find_all('td')
          if len(std_cols) < STD_P_FIELD_CNT:
            continue
          if 'partial_table' in std_row['class']:
            p_data[-1]['TM'] = get_data_from_cols(std_cols, 2) # Update the last player's team
            continue
          record_dict = {}
          # Standard pitching stats
          record_dict['PID'] = lastname_initial + str(player_cnt)
          record_dict['NAME'] = player_name.split()[-1] + ', ' + player_name.split()[0]
          record_dict['AGE'] = get_data_from_cols(std_cols, 1)
          record_dict['T'] = arm.encode('utf-8')[0]
          record_dict['YR'] = get_data_from_cols(std_cols, 0)
          record_dict['TM'] = get_data_from_cols(std_cols, 2)
          if 'minors_table' in std_row['class'] and not str(record_dict['TM']).endswith('-min'):
            record_dict['TM'] = str(record_dict['TM']) + '-min'
          record_dict['G'] = get_data_from_cols(std_cols, 8)
          record_dict['GS'] = get_data_from_cols(std_cols, 9)
          record_dict['W'] = get_data_from_cols(std_cols, 4)
          record_dict['L'] = get_data_from_cols(std_cols, 5)
          record_dict['WLP'] = get_data_from_cols(std_cols, 6, 3)
          record_dict['CG'] = get_data_from_cols(std_cols, 11)
          record_dict['ShO'] = get_data_from_cols(std_cols, 12)
          record_dict['IP'] = get_data_from_cols(std_cols, 14, 1)
          record_dict['H'] = get_data_from_cols(std_cols, 15)
          record_dict['R'] = get_data_from_cols(std_cols, 16)
          record_dict['ER'] = get_data_from_cols(std_cols, 17)
          record_dict['TBB'] = get_data_from_cols(std_cols, 19)
          record_dict['IBB'] = get_data_from_cols(std_cols, 20)
          record_dict['SO'] = get_data_from_cols(std_cols, 21)
          record_dict['ERA'] = get_data_from_cols(std_cols, 7, 2)
          record_dict['GF'] = get_data_from_cols(std_cols, 10)
          record_dict['SV'] = get_data_from_cols(std_cols, 13)
          record_dict['ShO'] = get_data_from_cols(std_cols, 13)
          record_dict['HR'] = get_data_from_cols(std_cols, 18)
          record_dict['HB'] = get_data_from_cols(std_cols, 22)
          record_dict['BK'] = get_data_from_cols(std_cols, 23)
          record_dict['WP'] = get_data_from_cols(std_cols, 24)
          record_dict['H/9'] = get_data_from_cols(std_cols, 29, 2)
          record_dict['SO/9'] = get_data_from_cols(std_cols, 32, 2)
          record_dict['BB/9'] = get_data_from_cols(std_cols, 31, 2)
          record_dict['HR/9'] = get_data_from_cols(std_cols, 30, 2)
          if record_dict['SO'] and record_dict['TBB'] and float(record_dict['TBB']) > 0.0:
            record_dict['SO/BB'] = fpformat.fix( (float(record_dict['SO']) / float(record_dict['TBB'])), 2 )
          else:
            record_dict['SO/BB'] = None
          if record_dict['IP'] and float(record_dict['IP']) > 0.0\
            and record_dict['H'] and record_dict['TBB'] and record_dict['HB']:
            record_dict['BR/9'] = fpformat.fix( (((float(record_dict['H']) + float(record_dict['TBB']) + float(record_dict['HB'])) / float(record_dict['IP'])) * 9) , 2)
          else:
            record_dict['BR/9'] = None
          # Pitcher values
          if record_dict['TM'].endswith('-min'):
            p_data.append(record_dict)
            pitcher_rec_cnt += 1
            continue
          val_rows = []
          for val_row in player_val_p_table:
            if 'blank_table' in val_row['class']:
              continue
            val_cols = val_row.find_all('td')
            if len(val_cols) < VAL_P_FIELD_CNT:
              continue
            val_yr = str(val_cols[0].contents[0]).encode('utf-8')
            if int(val_yr) < int(record_dict['YR']):
              continue
            elif int(val_yr) > int(record_dict['YR']):
              break
            else:
              val_rows.append(val_cols)
          if len(val_rows) == 1:
            record_dict['RAA'] = get_data_from_cols(val_rows[0], 14)
            record_dict['WAA'] = get_data_from_cols(val_rows[0], 15, 1)
            record_dict['WAAadj'] = get_data_from_cols(val_rows[0], 17, 1)
            record_dict['WAR'] = get_data_from_cols(val_rows[0], 18, 1)
            record_dict['RAR'] = get_data_from_cols(val_rows[0], 19)
            record_dict['waaWL%'] = get_data_from_cols(val_rows[0], 20, 3)
            record_dict['162WL%'] = get_data_from_cols(val_rows[0], 21, 3)
          else:
            record_dict['RAA'] = 0
            record_dict['WAA'] = 0
            record_dict['WAAadj'] = 0
            record_dict['WAR'] = 0
            record_dict['RAR'] = 0
            record_dict['waaWL%'] = 0
            record_dict['162WL%'] = 0
            for val_row in val_rows:
              gp = float(get_data_from_cols(val_row, 5))
              partial = gp / float(record_dict['G'])
              if partial >= 1:
                timedLog('partial error: ' + record_dict['NAME'] + ' ' + record_dict['YR'] + ' ' + str(gp) + ' ' + str(record_dict['G']))
              if get_data_from_cols(val_row, 14):
                record_dict['RAA'] = fpformat.fix(float(record_dict['RAA']) + float(get_data_from_cols(val_row, 14)) * partial, 0)
              if get_data_from_cols(val_row, 15, 1):
                record_dict['WAA'] = fpformat.fix(float(record_dict['WAA']) + float(get_data_from_cols(val_row, 15, 1)) * partial, 1)
              if get_data_from_cols(val_row, 17, 1):
                record_dict['WAAadj'] = fpformat.fix(float(record_dict['WAAadj']) + float(get_data_from_cols(val_row, 17, 1)) * partial, 1)
              if get_data_from_cols(val_row, 18, 1):
                record_dict['WAR'] = fpformat.fix(float(record_dict['WAR']) + float(get_data_from_cols(val_row, 18, 1)) * partial, 1)
              if get_data_from_cols(val_row, 19):
                record_dict['RAR'] = fpformat.fix(float(record_dict['RAR']) + float(get_data_from_cols(val_row, 19)) * partial, 0)
              if get_data_from_cols(val_row, 20, 3):
                record_dict['waaWL%'] = fpformat.fix(float(record_dict['waaWL%']) + float(get_data_from_cols(val_row, 20, 3)) * partial, 3)
              if get_data_from_cols(val_row, 21, 3):
                record_dict['162WL%'] = fpformat.fix(float(record_dict['162WL%']) + float(get_data_from_cols(val_row, 21, 3)) * partial, 3)
          p_data.append(record_dict)
          pitcher_rec_cnt += 1
      else:
        hitter_cnt += 1
        arm = player_positions_obj.next_sibling.next_sibling.next_sibling.next_element.next_element.split(',')[0].strip()
        player_std_h_table = player_page_obj.find('table', id='batting_standard').tbody.find_all('tr')
        player_val_h_table = player_page_obj.find('table', id='batting_value').tbody.find_all('tr')
        for std_row in player_std_h_table:
          if 'blank_table' in std_row['class']:
            continue
          std_cols = std_row.find_all('td')
          if len(std_cols) < STD_H_FIELD_CNT:
            continue
          if 'partial_table' in std_row['class']:
            h_data[-1]['TM'] = get_data_from_cols(std_cols, 2) # Update the last player's team
            continue
          record_dict = {}
          # Standard hitting stats
          record_dict['PID'] = lastname_initial + str(player_cnt)
          record_dict['NAME'] = player_name.split()[-1] + ', ' + player_name.split()[0]
          record_dict['AGE'] = get_data_from_cols(std_cols, 1)
          record_dict['B'] = arm.encode('utf-8')[0]
          if record_dict['B'] == 'S':
            record_dict['B'] = 'B'
          record_dict['YR'] = get_data_from_cols(std_cols, 0)
          record_dict['TM'] = get_data_from_cols(std_cols, 2)
          if 'minors_table' in std_row['class'] and not str(record_dict['TM']).endswith('-min'):
            record_dict['TM'] = str(record_dict['TM']) + '-min'
          record_dict['POS'] = get_position(get_data_from_cols(std_cols, 28))
          record_dict['G'] = get_data_from_cols(std_cols, 4)
          record_dict['PA'] = get_data_from_cols(std_cols, 5)
          record_dict['AB'] = get_data_from_cols(std_cols, 6)
          record_dict['R'] = get_data_from_cols(std_cols, 7)
          record_dict['H'] = get_data_from_cols(std_cols, 8)
          record_dict['2B'] = get_data_from_cols(std_cols, 9)
          record_dict['3B'] = get_data_from_cols(std_cols, 10)
          record_dict['HR'] = get_data_from_cols(std_cols, 11)
          record_dict['RBI'] = get_data_from_cols(std_cols, 12)
          if record_dict['R'] and record_dict['RBI']:
            record_dict['RP'] = int(record_dict['R']) + int(record_dict['RBI'])
          else:
            record_dict['RP'] = None
          record_dict['TBB'] = get_data_from_cols(std_cols, 15)
          record_dict['IBB'] = get_data_from_cols(std_cols, 27)
          record_dict['SO'] = get_data_from_cols(std_cols, 16)
          record_dict['SB'] = get_data_from_cols(std_cols, 13)
          record_dict['CS'] = get_data_from_cols(std_cols, 14)
          if record_dict['SB'] and record_dict['CS'] and (float(record_dict['SB']) + float(record_dict['CS'])) != 0:
            record_dict['SBP'] = fpformat.fix(float(record_dict['SB']) / (float(record_dict['SB']) + float(record_dict['CS'])), 3)
          else:
            record_dict['SBP'] = None
          record_dict['AVG'] = get_data_from_cols(std_cols, 17)
          record_dict['OBP'] = get_data_from_cols(std_cols, 18)
          record_dict['SLG'] = get_data_from_cols(std_cols, 19)
          record_dict['OPS'] = get_data_from_cols(std_cols, 20)
          record_dict['SH'] = get_data_from_cols(std_cols, 25)
          record_dict['SF'] = get_data_from_cols(std_cols, 26)
          record_dict['HBP'] = get_data_from_cols(std_cols, 24)
          record_dict['TB'] = get_data_from_cols(std_cols, 22)
          record_dict['GDP'] = get_data_from_cols(std_cols, 23)
          record_dict['OPS+'] = get_data_from_cols(std_cols, 21)
          if record_dict['H'] and record_dict['TBB'] and record_dict['TB'] and record_dict['AB'] and float(record_dict['AB']) + float(record_dict['TBB']) != 0:
            record_dict['RCbasic'] = (float(record_dict['H']) + float(record_dict['TBB'])) * float(record_dict['TB']) / (float(record_dict['AB']) + float(record_dict['TBB']))
          else:
            record_dict['RCbasic'] = None
          if record_dict['H'] and record_dict['TBB'] and record_dict['CS'] and record_dict['TB'] and record_dict['SB'] and (float(record_dict['AB']) + float(record_dict['TBB'])) != 0:
            record_dict['RCsb'] = (float(record_dict['H']) + float(record_dict['TBB']) - float(record_dict['CS'])) * (float(record_dict['TB']) + float(0.55 * float(record_dict['SB']))) / (float(record_dict['AB']) + float(record_dict['TBB']))
          else:
            record_dict['RCsb'] = None
          if record_dict['H'] and record_dict['TBB'] and record_dict['CS'] and record_dict['HBP'] and record_dict['GDP'] and record_dict['TB'] and record_dict['TBB'] and record_dict['IBB'] and record_dict['SH'] and record_dict['SF'] and record_dict['SB'] and record_dict['AB'] and (float(record_dict['AB']) + float(record_dict['TBB']) + float(record_dict['HBP']) + float(record_dict['SH']) + float(record_dict['SF'])) != 0:
            record_dict['RCtech'] = (float(record_dict['H']) + float(record_dict['TBB']) - float(record_dict['CS']) + float(record_dict['HBP']) - float(record_dict['GDP'])) * (float(record_dict['TB']) + float(0.26 * (float(record_dict['TBB']) - float(record_dict['IBB']) + float(record_dict['HBP'])))) + (0.52 * (float(record_dict['SH']) + float(record_dict['SF']) + float(record_dict['SB']))) / (float(record_dict['AB']) + float(record_dict['TBB']) + float(record_dict['HBP']) + float(record_dict['SH']) + float(record_dict['SF']))
          else:
            record_dict['RCtech'] = None
          # Hitter values
          if record_dict['TM'].endswith('-min'):
            h_data.append(record_dict)
            hitter_rec_cnt += 1
            continue
          val_rows = []
          for val_row in player_val_h_table:
            if 'blank_table' in val_row['class']:
              continue
            val_cols = val_row.find_all('td')
            if len(val_cols) < VAL_H_FIELD_CNT:
              continue
            val_yr = str(val_cols[0].contents[0]).encode('utf-8')
            if int(val_yr) < int(record_dict['YR']):
              continue
            elif int(val_yr) > int(record_dict['YR']):
              break
            else:
              val_rows.append(val_cols)
          if len(val_rows) == 1:
            record_dict['RAA'] = get_data_from_cols(val_rows[0], 11)
            record_dict['WAA'] = get_data_from_cols(val_rows[0], 12, 1)
            record_dict['RAR'] = get_data_from_cols(val_rows[0], 14)
            record_dict['WAR'] = get_data_from_cols(val_rows[0], 15, 1)
            record_dict['waaWL%'] = get_data_from_cols(val_rows[0], 16, 3)
            record_dict['162WL%'] = get_data_from_cols(val_rows[0], 17, 3)
            record_dict['oWAR'] = get_data_from_cols(val_rows[0], 18, 1)
            record_dict['dWAR'] = get_data_from_cols(val_rows[0], 19, 1)
            record_dict['oRAR'] = get_data_from_cols(val_rows[0], 20)
          else:
            record_dict['RAA'] = 0
            record_dict['WAA'] = 0
            record_dict['RAR'] = 0
            record_dict['WAR'] = 0
            record_dict['waaWL%'] = 0
            record_dict['162WL%'] = 0
            record_dict['oWAR'] = 0
            record_dict['dWAR'] = 0
            record_dict['oRAR'] = 0
            for val_row in val_rows:
              gp = float(get_data_from_cols(val_row, 4))
              partial = gp / float(record_dict['G'])
              if partial >= 1:
                timedLog('partial error: ' + record_dict['NAME'] + ' ' + record_dict['YR'] + ' ' + str(gp) + ' ' + str(record_dict['G']))
              if get_data_from_cols(val_row, 11, 0):
                record_dict['RAA'] = fpformat.fix(float(record_dict['RAA']) + float(get_data_from_cols(val_row, 11, 0)) * partial, 0)
              if get_data_from_cols(val_row, 12, 1):
                record_dict['WAA'] = fpformat.fix(float(record_dict['WAA']) + float(get_data_from_cols(val_row, 12, 1)) * partial, 1)
              if get_data_from_cols(val_row, 14):
                record_dict['RAR'] = fpformat.fix(float(record_dict['RAR']) + float(get_data_from_cols(val_row, 14)) * partial, 0)
              if get_data_from_cols(val_row, 15, 1):
                record_dict['WAR'] = fpformat.fix(float(record_dict['WAR']) + float(get_data_from_cols(val_row, 15, 1)) * partial, 1)
              if get_data_from_cols(val_row, 16, 3):
                record_dict['waaWL%'] = fpformat.fix(float(record_dict['waaWL%']) + float(get_data_from_cols(val_row, 16, 3)) * partial, 3)
              if get_data_from_cols(val_row, 17, 3):
                record_dict['162WL%'] = fpformat.fix(float(record_dict['162WL%']) + float(get_data_from_cols(val_row, 17, 3)) * partial, 3)
              if get_data_from_cols(val_row, 18, 1):
                record_dict['oWAR'] = fpformat.fix(float(record_dict['oWAR']) + float(get_data_from_cols(val_row, 18, 1)) * partial, 1)
              if get_data_from_cols(val_row, 19, 1):
                record_dict['dWAR'] = fpformat.fix(float(record_dict['dWAR']) + float(get_data_from_cols(val_row, 19, 1)) * partial, 1)
              if get_data_from_cols(val_row, 20):
                record_dict['oRAR'] = fpformat.fix(float(record_dict['oRAR']) + float(get_data_from_cols(val_row, 20)) * partial, 0)
          h_data.append(record_dict)
          hitter_rec_cnt += 1
      player_cnt += 1
  print '\n'
timedLog('processed ' + str(pitcher_cnt) + ' pitchers')
timedLog('processed ' + str(pitcher_rec_cnt) + ' pitcher records')
timedLog('processed ' + str(hitter_cnt) + ' hitters')
timedLog('processed ' + str(hitter_rec_cnt) + ' hitter records')
p_df = pandas.DataFrame.from_records(p_data, columns=p_header)
p_df.to_csv(p_file + target_initial + '.csv')
h_df = pandas.DataFrame.from_records(h_data, columns=h_header)
h_df.to_csv(h_file + target_initial + '.csv')
