'''
2023/12/18
特徴量生成コード
指文字を表現するデータを時間軸で4分割し、それぞれに対する特徴量を生成する
特徴生成の計算を関数化することで繰り返し処理するときに冗長性を回避し、フレームが4つしかない(range関数で実行できない)データにも適用できるようにする

feature_generator_func.pyの後継で、angleがx,y,zごとに出ていなかった可能性を修正

293行目の最後のフレームをframe_numからframe_num - 1に修正
'''
import os
import pandas as pd
from tqdm import tqdm
import numpy as np
import math
from natsort import natsorted
import copy
import pandas as pd

def feature_caluc(separate_num, start_index, end_index, distance_average, angle_average, thumb_direction_averages, index_finger_direction_averages, middle_finger_direction_averages, ring_finger_direction_averages, pinkie_finger_direction_averages):
    used_frame_counter = end_index - start_index

    # 各区切りで得られた特徴量を一時的に保存する変数
    distance_average_tmp = np.array([0 for j in range(190)], dtype = 'float')
    angle_average_tmp = np.array([0 for j in range(630)], dtype = 'float')
    thumb_direction = np.array([0,0,0], dtype = 'float')
    index_finger_direction = np.array([0,0,0], dtype = 'float')
    middle_finger_direction = np.array([0,0,0], dtype = 'float')
    ring_finger_direction = np.array([0,0,0], dtype = 'float')
    pinkie_finger_direction = np.array([0,0,0], dtype = 'float')

    for j in range(start_index, end_index):
        '''
        distanceの生成
        '''
        distance_index = 0
        for k in range(20):
            for l in range(k+1, 21):
                # 隣り合った指や手首のlandmarkならskip
                if(l-k==1) and k%4!=0 or (k==0 and(l==1 or l==5 or l==9 or l==13 or l==17)):
                    continue
                # 3次元の点と点同士の距離をdistance_average[1-2などに入れる]
                distance_average_tmp[distance_index] += math.sqrt((raw_data_list[j][k][0]-raw_data_list[j][l][0])**2 + (raw_data_list[j][k][1]-raw_data_list[j][l][1])**2 + (raw_data_list[j][k][2]-raw_data_list[j][l][2])**2)
                distance_index += 1

        '''
        angleの生成
        '''
        angle_index = 0
        for k in range(20):
            for l in range(k+1, 21):
                # x角度
                angle_average_tmp[angle_index] += math.acos((raw_data_list[j][l][0] - raw_data_list[j][k][0]) / (math.sqrt((raw_data_list[j][l][0]-raw_data_list[j][k][0])**2 + (raw_data_list[j][l][1]-raw_data_list[j][k][1])**2 + (raw_data_list[j][l][2]-raw_data_list[j][k][2])**2)))
                angle_index += 1
                # y角度
                angle_average_tmp[angle_index] += math.acos((raw_data_list[j][l][1] - raw_data_list[j][k][1]) / (math.sqrt((raw_data_list[j][l][0]-raw_data_list[j][k][0])**2 + (raw_data_list[j][l][1]-raw_data_list[j][k][1])**2 + (raw_data_list[j][l][2]-raw_data_list[j][k][2])**2)))
                angle_index += 1
                # z角度
                angle_average_tmp[angle_index] += math.acos((raw_data_list[j][l][2] - raw_data_list[j][k][2]) / (math.sqrt((raw_data_list[j][l][0]-raw_data_list[j][k][0])**2 + (raw_data_list[j][l][1]-raw_data_list[j][k][1])**2 + (raw_data_list[j][l][2]-raw_data_list[j][k][2])**2)))
                angle_index += 1
        
        '''
        directionの生成
        averageを求める。
        三次元単位ベクトル
        '''
        thumb_direction_tmp = np.array([0,0,0], dtype = 'float')
        index_finger_direction_tmp = np.array([0,0,0], dtype = 'float')
        middle_finger_direction_tmp = np.array([0,0,0], dtype = 'float')
        ring_finger_direction_tmp = np.array([0,0,0], dtype = 'float')
        pinkie_finger_direction_tmp = np.array([0,0,0], dtype = 'float')
        # 親指 1 -> 4
        # 指先 - 指元でベクトルを計算し、その後で単位ベクトルに変換
        thumb_direction_tmp[0] = raw_data_list[j][4][0] - raw_data_list[j][1][0]
        thumb_direction_tmp[1] = raw_data_list[j][4][1] - raw_data_list[j][1][1]
        thumb_direction_tmp[2] = raw_data_list[j][4][2] - raw_data_list[j][1][2]
        thumb_vector_size = math.sqrt((thumb_direction_tmp[0] ** 2) + (thumb_direction_tmp[1] ** 2) + (thumb_direction_tmp[2] ** 2))  # 単位ベクトル計算のためベクトルの大きさを出す
        thumb_direction_tmp[0] /= thumb_vector_size
        thumb_direction_tmp[1] /= thumb_vector_size
        thumb_direction_tmp[2] /= thumb_vector_size
        # 単位ベクトルが出たので、時間範囲で平均する用の変数に足す
        thumb_direction += thumb_direction_tmp

        # 人差し指 5 -> 8
        index_finger_direction_tmp[0] = raw_data_list[j][8][0] - raw_data_list[j][5][0]
        index_finger_direction_tmp[1] = raw_data_list[j][8][1] - raw_data_list[j][5][1]
        index_finger_direction_tmp[2] = raw_data_list[j][8][2] - raw_data_list[j][5][2]
        index_finger_vector_size = math.sqrt((index_finger_direction_tmp[0] ** 2) + (index_finger_direction_tmp[1] ** 2) + (index_finger_direction_tmp[2] ** 2))  # 単位ベクトル計算のためベクトルの大きさを出す
        index_finger_direction_tmp[0] /= index_finger_vector_size
        index_finger_direction_tmp[1] /= index_finger_vector_size
        index_finger_direction_tmp[2] /= index_finger_vector_size
        # 単位ベクトルが出たので、時間範囲で平均する用の変数に足す
        index_finger_direction += index_finger_direction_tmp

        # 中指 9 -> 12
        middle_finger_direction_tmp[0] = raw_data_list[j][12][0] - raw_data_list[j][9][0]
        middle_finger_direction_tmp[1] = raw_data_list[j][12][1] - raw_data_list[j][9][1]
        middle_finger_direction_tmp[2] = raw_data_list[j][12][2] - raw_data_list[j][9][2]
        middle_finger_vector_size = math.sqrt((middle_finger_direction_tmp[0] ** 2) + (middle_finger_direction_tmp[1] ** 2) + (middle_finger_direction_tmp[2] ** 2))  # 単位ベクトル計算のためベクトルの大きさを出す
        middle_finger_direction_tmp[0] /= middle_finger_vector_size
        middle_finger_direction_tmp[1] /= middle_finger_vector_size
        middle_finger_direction_tmp[2] /= middle_finger_vector_size
        # 単位ベクトルが出たので、時間範囲で平均する用の変数に足す
        middle_finger_direction += middle_finger_direction_tmp

        # 薬指　13 -> 16
        ring_finger_direction_tmp[0] = raw_data_list[j][16][0] - raw_data_list[j][13][0]
        ring_finger_direction_tmp[1] = raw_data_list[j][16][1] - raw_data_list[j][13][1]
        ring_finger_direction_tmp[2] = raw_data_list[j][16][2] - raw_data_list[j][13][2]
        ring_finger_vector_size = math.sqrt((ring_finger_direction_tmp[0] ** 2) + (ring_finger_direction_tmp[1] ** 2) + (ring_finger_direction_tmp[2] ** 2))  # 単位ベクトル計算のためベクトルの大きさを出す
        ring_finger_direction_tmp[0] /= ring_finger_vector_size
        ring_finger_direction_tmp[1] /= ring_finger_vector_size
        ring_finger_direction_tmp[2] /= ring_finger_vector_size
        # 単位ベクトルが出たので、時間範囲で平均する用の変数に足す
        ring_finger_direction += ring_finger_direction_tmp

        # 小指 17 -> 20
        pinkie_finger_direction_tmp[0] = raw_data_list[j][20][0] - raw_data_list[j][17][0]
        pinkie_finger_direction_tmp[1] = raw_data_list[j][20][1] - raw_data_list[j][17][1]
        pinkie_finger_direction_tmp[2] = raw_data_list[j][20][2] - raw_data_list[j][17][2]
        pinkie_finger_vector_size = math.sqrt((pinkie_finger_direction_tmp[0] ** 2) + (pinkie_finger_direction_tmp[1] ** 2) + (pinkie_finger_direction_tmp[2] ** 2))  # 単位ベクトル計算のためベクトルの大きさを出す
        pinkie_finger_direction_tmp[0] /= pinkie_finger_vector_size
        pinkie_finger_direction_tmp[1] /= pinkie_finger_vector_size
        pinkie_finger_direction_tmp[2] /= pinkie_finger_vector_size
        # 単位ベクトルが出たので、時間範囲で平均する用の変数に足す
        pinkie_finger_direction += pinkie_finger_direction_tmp
    
    
    if used_frame_counter == 0:
        print("0で割るパターンが実行")
        print("hand_type: " + str(i))
        print("frame_num: " + str(frame_num))
        print("split_num: " + str(split_num))
        print(start_indexes)
    #今まで計算した特徴はすべてのフレームの合計なので、使用　フレーム数で割って平均にする
    for j in range(len(distance_average_tmp)):
        distance_average_tmp[j] = distance_average_tmp[j] / used_frame_counter
    for j in range(len(angle_average_tmp)):
        angle_average_tmp[j] = angle_average_tmp[j] / used_frame_counter
    
    for j in range(3):
        thumb_direction[j] /= used_frame_counter
        index_finger_direction[j] /= used_frame_counter
        middle_finger_direction[j] /= used_frame_counter
        ring_finger_direction[j] /= used_frame_counter
        pinkie_finger_direction[j] /= used_frame_counter
    
    # データ確認
    # confirm_variable_name = variation_tmp
    # print("データ確認")
    # print(len(confirm_variable_name))
    # print(confirm_variable_name)

    '''
    distance_average_tmp: 190個の実数データ
    angle_average:        630個の実数データ
    thumb_direction:      
    '''
    # この分割域で計算した特徴を一つの変数にまとめる
    distance_average = np.append(distance_average, distance_average_tmp)
    angle_average = np.append(angle_average, angle_average_tmp)
    thumb_direction_averages = np.append(thumb_direction_averages, thumb_direction)
    index_finger_direction_averages = np.append(index_finger_direction_averages, index_finger_direction)
    middle_finger_direction_averages = np.append(middle_finger_direction_averages, middle_finger_direction)
    ring_finger_direction_averages = np.append(ring_finger_direction_averages,ring_finger_direction)
    pinkie_finger_direction_averages = np.append(pinkie_finger_direction_averages, pinkie_finger_direction)


    return distance_average, angle_average, thumb_direction_averages, index_finger_direction_averages, middle_finger_direction_averages, ring_finger_direction_averages, pinkie_finger_direction_averages



STATIC_FILE_USE_NUM = 12  # 静的指文字の特徴量生成に使用するフレーム数
START_INDEX = 5  # 静的指文字の特徴量生成に使用するフレームの開始位置

Name = input("特徴抽出するディレクトリ名を入力: ")
srcdir = os.environ['HOME']+'/lab-past/kakizaki/data/' + str(Name) + '/mediapipe_csv/'
wrtdir = './features/4divide-features-final-frame-fixed/'

try:
    os.makedirs(wrtdir)
except FileExistsError:
    print('ディレクトリ(' + str(wrtdir) + ')はすでに存在しているため作成しません')
    pass

file = os.listdir(srcdir)  # srcdirのファイル名をリストですべて持って来る
srcFiles = natsorted(file)  # ファイル名順にソート(この場合数字でソートされるのでlabel順、フレーム順になる)

'''
ここからデータをそれぞれのhand_typeごとのリストにする処理
データ格納先を宣言 -> ファイル名で分けてそれぞれのhand_typeに分ける
'''
hand_data_list = [[] for i in range(47)]

for f in srcFiles:
    hand_type = int(f.split('_')[1])
    hand_data_list[hand_type].append(f)

'''
hand_dataごとに分けてたリストhand_data_listのデータ数を調べることで、
最もフレーム数の少ないhand_typeのフレーム数を取得する
これらは時間軸データ分割の4分割に耐えられるかを確認するため
'''
minDataNum = 4
minDataHandType = []
for i in range(1,47):
    if len(hand_data_list[i]) < minDataNum:
        minDataNum = len(hand_data_list[i])
        minDataHandType.append(i)

# もし最小のフレーム数が4を下回る場合にはその旨を通知し、処理を続行するかどうか決める
if minDataNum < 4:
    print(str(minDataHandType) + "でデータ数が足りませんでした。")
    user_input = input("データ数が4を下回り分割が行えません。処理を続けますか? y/n : ")
    if user_input == 'y':
        print("処理を続行します")
    else:
        print("データ数が十分にないため特徴量生成を中止します")
        exit(1)


'''
静的指文字のフレーム数を動的指文字のフレーム数に合わせる処理を行う
静的指文字の特徴抽出に使用するフレーム数は動的指文字の平均フレーム数から算出され、
コード上部のSTATIC_FILE_USE_NUMに定義される
使用フレームはだいたい真ん中にしたいが、そのスタート位置はコード上部START_INDEXに定義する
'''
error_hand_type = []
for i in range(1,42):  # 静的指文字のファイル名を12個に減らす
    new_file_list = []
    if len(hand_data_list[i]) <= 12:
        j = 0
    else:
        j = START_INDEX  # カウンタ変数
    while len(new_file_list) < STATIC_FILE_USE_NUM:  # 規定ファイル数を確保するまで
        try:
            new_file_list.append(hand_data_list[i][j])
        except:
            print("静的指文字のファイル数が足りませんでした")
            print("hand_type: " + str(i))
            print("ファイル箇所: " + str(j))
            #exit(1)
            #ここで強制終了せずに12以下のファイル数でも実行する
            if len(new_file_list) < 4:
                print("静的指文字のファイル数が4を下回るため処理を停止します")
                error_hand_type.append(i)
                print("静的指文字のファイル数が少ないものをerror_hand_typeリストに追加しました")
                # exit(1)
                print("例外的に継続します")
                break
            else:
                print("静的指文字のファイル数が少ないものをerror_hand_typeリストに追加しました")
                error_hand_type.append(i)
                break
        j = j + 1
    # new_file_listに置き換える
    hand_data_list[i] = copy.deepcopy(new_file_list)

'''
ここから実際に特徴抽出を行っていく
'''
# まずはファイル名リストhand_data_listを使って実際のデータをnumpy配列に取ってくる

for i in tqdm(range(1,len(hand_data_list))):
    #データ数が4を下回った場合に、そのhand_typeは特徴計算をしない
    if i in minDataHandType:
        continue
    if i in error_hand_type:
        continue

    frame_num = len(hand_data_list[i])
    split_num = frame_num / 4  # データを時間軸で4分割するため、何個ずつにするか
    split_remainder = frame_num % 4 # 余りの数によって一区切りの枚数に+1することで全てのフレームを使うようにする
    raw_data_list = []
    for j in range(len(hand_data_list[i])):
        # データ読み込み
        data = np.loadtxt(srcdir + hand_data_list[i][j], delimiter=',')
        raw_data_list.append(data)
    
    # print(raw_data_list)
    # ここまででraw_data_listに座標データ、frame_numにそのhand_typeのフレーム数が入っている
    start_indexes = [0]
    for j in range(3):
        if j == 3:
            start_indexes.append(int(start_indexes[j] + split_num))
        else:
            # これで繰り上げるパターンにできるが繰り上げたくないためコメントアウト
            # start_indexes.append(math.ceil(start_indexes[j] + split_num))
            if split_remainder > 0:
                split_remainder -= 1
                start_indexes.append(int(start_indexes[j] + split_num + 1))
            else:
                start_indexes.append(int(start_indexes[j] + split_num))
    
    start_indexes.append(frame_num-1)
    # print("start_indexes: " + str(start_indexes))
    # start_indexesに入っている数字から次の数字までの間で一つの特徴量を生成する
    '''
    Distance: average
    Angle: average
    Variation: 最初のフレームと最後のフレームの移動ベクトル
    Direction: average, 各指が向いている方向を方向ベクトルで示す
    '''
    all_columns = []
    all_features = np.array([], dtype = 'float')
    distance_average = np.array([], dtype = 'float')
    angle_average = np.array([], dtype = 'float')
    variation = np.array([], dtype = 'float')
    thumb_direction_averages = np.array([], dtype = 'float')
    index_finger_direction_averages = np.array([], dtype = 'float')
    middle_finger_direction_averages = np.array([], dtype = 'float')
    ring_finger_direction_averages = np.array([], dtype = 'float')
    pinkie_finger_direction_averages = np.array([], dtype = 'float')

    # raw_data_list[0 -> 1/4]
    distance_average, angle_average, thumb_direction_averages, index_finger_direction_averages, middle_finger_direction_averages, ring_finger_direction_averages, pinkie_finger_direction_averages = feature_caluc('1in4', start_indexes[0], start_indexes[1], distance_average, angle_average, thumb_direction_averages, index_finger_direction_averages, middle_finger_direction_averages, ring_finger_direction_averages, pinkie_finger_direction_averages)
    # variation 1個目(全3個)
    for j in range(21):
        # mark3.append("1of4_variationX_" + str(j+1) )
        variation = np.append(variation, raw_data_list[start_indexes[1]][j][0] - raw_data_list[start_indexes[0]][j][0])  # x座標の最初のフレームと最後のフレームの差
        # mark3.append("1of4_variationY_" + str(j+1) )
        variation = np.append(variation, raw_data_list[start_indexes[1]][j][1] - raw_data_list[start_indexes[0]][j][1])  # y座標
        # mark3.append("1of4_variationZ_" + str(j+1) )
        variation = np.append(variation, raw_data_list[start_indexes[1]][j][2] - raw_data_list[start_indexes[0]][j][2])  # z座標

    # raw_data_list[1/4 -> 2/4]
    distance_average, angle_average, thumb_direction_averages, index_finger_direction_averages, middle_finger_direction_averages, ring_finger_direction_averages, pinkie_finger_direction_averages = feature_caluc('2in4', start_indexes[1], start_indexes[2], distance_average, angle_average, thumb_direction_averages, index_finger_direction_averages, middle_finger_direction_averages, ring_finger_direction_averages, pinkie_finger_direction_averages)
    # variation 2個目(全3個)
    for j in range(21):
        # mark3.append("1of4_variationX_" + str(j+1) )
        variation = np.append(variation, raw_data_list[start_indexes[2]][j][0] - raw_data_list[start_indexes[1]][j][0])  # x座標の最初のフレームと最後のフレームの差
        # mark3.append("1of4_variationY_" + str(j+1) )
        variation = np.append(variation, raw_data_list[start_indexes[2]][j][1] - raw_data_list[start_indexes[1]][j][1])  # y座標
        # mark3.append("1of4_variationZ_" + str(j+1) )
        variation = np.append(variation, raw_data_list[start_indexes[2]][j][2] - raw_data_list[start_indexes[1]][j][2])  # z座標

    # raw_data_list[2/4 -> 3/4]
    distance_average, angle_average, thumb_direction_averages, index_finger_direction_averages, middle_finger_direction_averages, ring_finger_direction_averages, pinkie_finger_direction_averages = feature_caluc('3in4', start_indexes[2], start_indexes[3], distance_average, angle_average, thumb_direction_averages, index_finger_direction_averages, middle_finger_direction_averages, ring_finger_direction_averages, pinkie_finger_direction_averages)
    # variation 一個目(全3個)
    for j in range(21):
        # mark3.append("1of4_variationX_" + str(j+1) )
        try:
            variation = np.append(variation, raw_data_list[start_indexes[3]][j][0] - raw_data_list[start_indexes[2]][j][0])  # x座標の最初のフレームと最後のフレームの差
        except:
            print("hand_type: " + str(i))
            print("split_num: " + str(split_num) )
            print("start_indexes: " + str(start_indexes))
        # mark3.append("1of4_variationY_" + str(j+1) )
        variation = np.append(variation, raw_data_list[start_indexes[3]][j][1] - raw_data_list[start_indexes[2]][j][1])  # y座標
        # mark3.append("1of4_variationZ_" + str(j+1) )
        variation = np.append(variation, raw_data_list[start_indexes[3]][j][2] - raw_data_list[start_indexes[2]][j][2])  # z座標

    # raw_data_list[3/4 -> 4/4]
    distance_average, angle_average, thumb_direction_averages, index_finger_direction_averages, middle_finger_direction_averages, ring_finger_direction_averages, pinkie_finger_direction_averages = feature_caluc('4in4', start_indexes[3], start_indexes[4], distance_average, angle_average, thumb_direction_averages, index_finger_direction_averages, middle_finger_direction_averages, ring_finger_direction_averages, pinkie_finger_direction_averages)
    # variationは3つまでしか生成できない


    #print("ここからファイルに書き込む")
    #計算した特徴をファイルに書き込むために一つの変数にまとめる
    all_features = np.array([], dtype="float")
    all_features = np.append(all_features, distance_average)
    all_features = np.append(all_features, angle_average)
    all_features = np.append(all_features, thumb_direction_averages)
    all_features = np.append(all_features, index_finger_direction_averages)
    all_features = np.append(all_features, middle_finger_direction_averages)
    all_features = np.append(all_features, ring_finger_direction_averages)
    all_features = np.append(all_features, pinkie_finger_direction_averages)
    all_features = np.append(all_features, variation)
    # print("hand_type: " + str(i))
    # print("frame_num: " + str(frame_num))
    # print("split_num: " + str(split_num))
    # print("start_indexes: " + str(start_indexes))

    # clumnの文字列リストを作る
    distance_columns = []
    for k in range(20):
            for l in range(k+1, 21):
                # 隣り合った指や手首のlandmarkならskip
                if(l-k==1) and k%4!=0 or (k==0 and(l==1 or l==5 or l==9 or l==13 or l==17)):
                    continue
                distance_columns.append('1in4' + 'distance_average' + str(k) + '-' + str(l))
    for k in range(20):
            for l in range(k+1, 21):
                # 隣り合った指や手首のlandmarkならskip
                if(l-k==1) and k%4!=0 or (k==0 and(l==1 or l==5 or l==9 or l==13 or l==17)):
                    continue
                distance_columns.append('2in4' + 'distance_average' + str(k) + '-' + str(l))
    for k in range(20):
            for l in range(k+1, 21):
                # 隣り合った指や手首のlandmarkならskip
                if(l-k==1) and k%4!=0 or (k==0 and(l==1 or l==5 or l==9 or l==13 or l==17)):
                    continue
                distance_columns.append('3in4' + 'distance_average' + str(k) + '-' + str(l))
    for k in range(20):
            for l in range(k+1, 21):
                # 隣り合った指や手首のlandmarkならskip
                if(l-k==1) and k%4!=0 or (k==0 and(l==1 or l==5 or l==9 or l==13 or l==17)):
                    continue
                distance_columns.append('4in4' + 'distance_average' + str(k) + '-' + str(l))
    

    # angleのclumnを作成する
    angle_columns = []
    for k in range(20):
            for l in range(k+1, 21):
                angle_columns.append('1in4' + 'X_angle_average' + str(k) + '-' + str(l))
                angle_columns.append('1in4' + 'Y_angle_average' + str(k) + '-' + str(l))
                angle_columns.append('1in4' + 'Z_angle_average' + str(k) + '-' + str(l))
    for k in range(20):
            for l in range(k+1, 21):
                angle_columns.append('2in4' + 'X_angle_average' + str(k) + '-' + str(l))
                angle_columns.append('2in4' + 'Y_angle_average' + str(k) + '-' + str(l))
                angle_columns.append('2in4' + 'Z_angle_average' + str(k) + '-' + str(l))
    for k in range(20):
            for l in range(k+1, 21):
                angle_columns.append('3in4' + 'X_angle_average' + str(k) + '-' + str(l))
                angle_columns.append('3in4' + 'Y_angle_average' + str(k) + '-' + str(l))
                angle_columns.append('3in4' + 'Z_angle_average' + str(k) + '-' + str(l))
    for k in range(20):
            for l in range(k+1, 21):
                angle_columns.append('4in4' + 'X_angle_average' + str(k) + '-' + str(l))
                angle_columns.append('4in4' + 'Y_angle_average' + str(k) + '-' + str(l))
                angle_columns.append('4in4' + 'Z_angle_average' + str(k) + '-' + str(l))

    # finger_directionのcolumnを作成
    finger_columns = []
    # 親指
    for j in range(4):
        finger_columns.append(str(j+1) + 'in4' + 'thumb_direction_x')
        finger_columns.append(str(j+1) + 'in4' + 'thumb_direction_y')
        finger_columns.append(str(j+1) + 'in4' + 'thumb_direction_z')
    # 人差し指
    for j in range(4):
        finger_columns.append(str(j+1) + 'in4' + 'index_direction_x')
        finger_columns.append(str(j+1) + 'in4' + 'index_direction_y')
        finger_columns.append(str(j+1) + 'in4' + 'index_direction_z')
    # 中指
    for j in range(4):
        finger_columns.append(str(j+1) + 'in4' + 'middle_direction_x')
        finger_columns.append(str(j+1) + 'in4' + 'middle_direction_y')
        finger_columns.append(str(j+1) + 'in4' + 'middle_direction_z')
    # 薬指
    for j in range(4):
        finger_columns.append(str(j+1) + 'in4' + 'ring_direction_x')
        finger_columns.append(str(j+1) + 'in4' + 'ring_direction_y')
        finger_columns.append(str(j+1) + 'in4' + 'ring_direction_z')
    # 小指
    for j in range(4):
        finger_columns.append(str(j+1) + 'in4' + 'pinkie_direction_x')
        finger_columns.append(str(j+1) + 'in4' + 'pinkie_direction_y')
        finger_columns.append(str(j+1) + 'in4' + 'pinkie_direction_z')
    

    # variationのcolumnを作成
    variation_columns = []
    for j in range(21):
        variation_columns.append('1to2' + 'variation[' + str(j) + ']:x')
        variation_columns.append('1to2' + 'variation[' + str(j) + ']:y')
        variation_columns.append('1to2' + 'variation[' + str(j) + ']:z')
    for j in range(21):
        variation_columns.append('2to3' + 'variation[' + str(j) + ']:x')
        variation_columns.append('2to3' + 'variation[' + str(j) + ']:y')
        variation_columns.append('2to3' + 'variation[' + str(j) + ']:z')
    for j in range(21):
        variation_columns.append('3to4' + 'variation[' + str(j) + ']:x')
        variation_columns.append('3to4' + 'variation[' + str(j) + ']:y')
        variation_columns.append('3to4' + 'variation[' + str(j) + ']:z')
    
    all_columns.extend(distance_columns)
    all_columns.extend(angle_columns)
    all_columns.extend(finger_columns)
    all_columns.extend(variation_columns)

    # print("columnsの数: " + str(len(all_columns)))
    df = pd.DataFrame(data=[all_features],columns=[all_columns])
    df.to_csv(wrtdir + Name + '_' + str(i) + '_' + 'feature.csv', index=False)
    # print(all_features)
    #np.savetxt(wrtdir + Name + '_' + str(i) + '_' + 'feature.csv', all_features, delimiter=',')
    # print(wrtdir + Name + '_' + str(i) + '_' + 'feature.csv')

print(str(Name) + "の特徴中ファイル生成が完了しました。")
print("ファイル数不足による生成失敗hand_type: " + str(minDataHandType))

print("エラーによって生成できなかったhand_type: " + str(error_hand_type))