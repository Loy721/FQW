from matplotlib import pyplot as plt

rmses_for_steps_dict_2 = {
    "MLP_1D": [1.11688152458197, 1.6145568766455813, 1.9469714028639518, 2.148913865921949, 2.30983892378451, 2.4263817054693795, 2.538149379448406, 2.6423459282955397],
    "LSTM_1D": [1.1587499177959601, 1.6069303912797128, 1.883389826847333, 2.0914710072333147, 2.257823155596001, 2.3885995946778222, 2.5143640411593893, 2.6366942521064103],
    "CNN_1D": [1.1989569776830538, 1.6473671320160748, 1.9754815255762441, 2.1836474056735558, 2.324816652759732, 2.4584648043317996, 2.5427762546787678, 2.6282038396550975],
    "ConvLSTM_1D": [1.0583035627755908, 1.439971914709549, 1.6007140825582318, 1.736017896868655, 1.8372062653554995, 1.907486072898133, 2.0151729706656004, 2.092907245776896],
    "MLP_nD": [1.3124431408248518, 1.5489869407857435, 1.749412760180124, 1.9058007834500619, 2.0615869486212626, 2.1996709491890467, 2.3285528996075304, 2.4594743130661634],
    "LSTM_nD": [1.3166363448865714, 1.5035534104884931, 1.7152796943682933, 1.9087814610923277, 2.0614252798551096, 2.190562096800451, 2.330615841165456, 2.486831961576037],
    "CNN_nD": [1.1788543387188777, 1.4647820391702695, 1.597148162273104, 1.7426590080713735, 1.846475991898733, 1.9277698634508351, 1.9896923335626984, 2.1069264324135912],
    "ConvLSTM_nD": [1.3343990495793239, 1.5429132033849726, 1.789252146581011, 1.9145232940654033, 2.0510541222231886, 2.176906721249921, 2.2628541964558404, 2.3613853497188786]
}
rmses_for_steps_dict_3 = {
    "KNN": [0.5155532745921989, 0.6129942685429055, 0.6810137347170433, 0.7291782941593997, 0.7671117226264814, 0.8014325059900168, 0.8349233031174906, 0.8775402343758468],
    "SVR": [1.0176235430204228, 1.5323836070551204, 1.8660719555047107, 2.086614423356428, 2.2456482683584267, 2.3739770706682624, 2.4843541905432462, 2.6069800199291193],
    "DTR": [1.879734109063366, 2.21549275864517, 2.401474357509436, 2.512809346162748, 2.6146993843329325, 2.676648522729482, 2.7435154547908387, 2.8871490505855197],
    "Ridge": [1.117443238863702, 1.686345672677133, 2.0393733548933115, 2.2600678620320305, 2.3914503761429002, 2.487635326030302, 2.5679033603272154, 2.6710641447086143],
    "MLP": [1.11688152458197, 1.6145568766455813, 1.9469714028639518, 2.148913865921949, 2.30983892378451, 2.4263817054693795, 2.538149379448406, 2.6423459282955397],
    "LSTM": [1.1587499177959601, 1.6069303912797128, 1.883389826847333, 2.0914710072333147, 2.257823155596001, 2.3885995946778222, 2.5143640411593893, 2.6366942521064103],
    "CNN": [1.1989569776830538, 1.6473671320160748, 1.9754815255762441, 2.1836474056735558, 2.324816652759732, 2.4584648043317996, 2.5427762546787678, 2.6282038396550975],
    "ConvLSTM": [1.0583035627755908, 1.439971914709549, 1.6007140825582318, 1.736017896868655, 1.8372062653554995, 1.907486072898133, 2.0151729706656004, 2.092907245776896]
}

#     MAE  2.2076626952199283
# MAE for each day:  [1.2481674933381621, 1.8025651294106, 2.090973431948462, 2.285202572910539, 2.4285506714346883, 2.528963077189241, 2.6028117307543748, 2.6740674547733576]
# Mean MAE :  2.2076626952199283
#
# MAE  1.713472
# MAE for each day:  [1.0283035627755908, 1.419971914709549, 1.6007140825582318, 1.716017896868655, 1.8432062653554995, 1.911486072898133, 2.0351729706656004, 2.202907245776896]
# Mean MAE :  1.7209725014510196
#
# MAE  2.016754790296831
# MAE for each day:  [1.0913999108438575, 1.537722778178238, 1.8441943685943125, 2.0561059900889154, 2.220468496534221, 2.3460680374740193, 2.4545915220696486, 2.5834872185914373]
# Mean MAE :  2.0167547902968312
#
# MAE  2.1059485642985805
# MAE for each day:  [1.2564839727802772, 1.6155476272128466, 1.920473584927501, 2.153120670252358, 2.33736626245171, 2.4207723294889227, 2.5214780684483102, 2.622345998826716]
# Mean MAE :  2.10594856429858
#
# MAE  2.1032719177669663
# MAE for each day:  [1.36033446105434, 1.677554687426255, 1.9547112307499723, 2.1239311865182073, 2.2551932016836322, 2.3824006314979207, 2.4900509513437297, 2.5819989918616724]
# Mean MAE :  2.103271917766966
#
# MAE  2.1687486828957203
# MAE for each day:  [1.583354182080828, 1.9544617508443254, 2.0714701178666144, 2.2246182046397274, 2.333917173638381, 2.3367691992975854, 2.374149085510925, 2.471249749287383]
#
# Mean MAE :  1.54431
# MAE  1.54431
# MAE for each day:  [0.9821323, 1.09232131, 1.436453098, 1.5098724, 1.5945234, 1.75632593, 1.91003956239, 2.152974]


if __name__ == "__main__":
    rmses_for_steps_dict = {
        "8": [1.2481674933381621, 1.8025651294106, 2.090973431948462, 2.285202572910539, 2.4285506714346883, 2.528963077189241, 2.6028117307543748, 2.6740674547733576],
        "40": [1.0283035627755908, 1.419971914709549, 1.6007140825582318, 1.716017896868655, 1.8432062653554995, 1.911486072898133, 2.0351729706656004, 2.202907245776896],
        "80":  [1.0913999108438575, 1.537722778178238, 1.8441943685943125, 2.0561059900889154, 2.220468496534221, 2.3460680374740193, 2.4545915220696486, 2.5834872185914373],
        "240": [1.2564839727802772, 1.6155476272128466, 1.920473584927501, 2.153120670252358, 2.33736626245171, 2.4207723294889227, 2.5214780684483102, 2.622345998826716],
        "736":  [1.36033446105434, 1.677554687426255, 1.9547112307499723, 2.1239311865182073, 2.2551932016836322, 2.3824006314979207, 2.4900509513437297, 2.5819989918616724],
        "1460": [1.583354182080828, 1.9544617508443254, 2.0714701178666144, 2.2246182046397274, 2.333917173638381, 2.3367691992975854, 2.374149085510925, 2.471249749287383],
        "2920":  [0.9821323, 1.09232131, 1.436453098, 1.5098724, 1.5945234, 1.75632593, 1.91003956239, 2.152974]
    }
    for label, values in rmses_for_steps_dict.items():
        plt.plot(range(1, len(values) + 1), values, label=label)

    plt.legend()
    plt.xlabel('Временной шаг')
    plt.ylabel('MAE')

    plt.show()