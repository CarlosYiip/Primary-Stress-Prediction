import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
import pickle

def is_vowel(v):
    return '0' in v or '1' in v or '2' in v

def has_v_as_ith_vowel_syllable(p, v, i):
    vowels_in_p = [j[:-1] for j in p.split(' ') if is_vowel(j)]
    if i > len(vowels_in_p):
        return False
    return v == vowels_in_p[i-1]

def n_syllable_before_ith_vowel_syllable(p, c, i, n):
    l = p.split(' ')
    count = 0
    for j in range(len(l)):
        if is_vowel(l[j]):
            count += 1
            if count == i:
                if j <= n-1:
                    return False
                else:
                    ans = l[j-n]
                    if is_vowel(ans):
                        return ans[:-1] == c
                    else:
                        return ans == c
    return False

def n_syllable_after_ith_vowel_syllable(p, c, i, n):
    l = p.split(' ')
    count = 0
    for j in range(len(l)):
        if is_vowel(l[j]):
            count += 1
            if count == i:
                if j >= len(l) - n:
                    return False
                else:
                    ans = l[j+n]
                    if is_vowel(ans):
                        return ans[:-1] == c
                    else:
                        return ans == c
                
    return False

def find_stress(p):
    l = [syllable for syllable in p.split(' ') if is_vowel(syllable)]
    for i in range(len(l)):
        if '1' in l[i]:
            return i + 1


def train(data, classifier_file):
    vowel_list = ['AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'EH', 'ER', 'EY', 'IH', 'IY', 'OW', 'OY', 'UH', 'UW']
    consonant_list = ['P', 'B', 'CH', 'D', 'DH', 'F', 'G', 'HH', 'JH', 'K', 'L', 'M', 'N', 'NG', 'R', 'S',
                  'SH', 'T', 'TH', 'V', 'W', 'Y', 'Z', 'ZH']

    syllable_list = vowel_list + consonant_list

    words, pronounciations = [], []
    df = pd.DataFrame()
##    data = open(data)
    for line in data:
        w, p = line.split(':')
        p = p.rstrip('\n')
        words.append(w)
        pronounciations.append(p)

    df['w'] = words
    df['p'] = pronounciations

    for v in vowel_list:
        for i in range(1, 5):
            df[v+str(i)] = df['p'].apply(lambda p: has_v_as_ith_vowel_syllable(p, v, i))
        
    for s in syllable_list:
        for i in range(1, 5):
            for j in range(1, 3):
                name = str(s) + str(j) + 'before' + str(i) + 'vowel syllable'
                df[name] = df['p'].apply(lambda p: n_syllable_before_ith_vowel_syllable(p, s, i, j))

    for s in syllable_list:
        for i in range(1, 5):
            for j in range(1, 3):
                name = str(s) + str(j) + 'after' + str(i) + 'vowel syllable'
                df[name] = df['p'].apply(lambda p: n_syllable_after_ith_vowel_syllable(p, s, i, j))           

    df['2'] = df['p'].apply(lambda p: len([i for i in p.split(' ') if is_vowel(i)]) == 2)
    df['3'] = df['p'].apply(lambda p: len([i for i in p.split(' ') if is_vowel(i)]) == 3)
    df['4'] = df['p'].apply(lambda p: len([i for i in p.split(' ') if is_vowel(i)]) == 4)
        
    df['stress'] = df['p'].apply(find_stress)



    clf = LogisticRegression(class_weight={4:1000})
    clf.fit(df[df.columns[2:-1]], df[df.columns[-1]])
    pickle.dump(clf, open(classifier_file, 'wb'))
    
    
def test(data, classifier_file):
    clf = pickle.load(open(classifier_file, 'rb'))

    vowel_list = ['AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'EH', 'ER', 'EY', 'IH', 'IY', 'OW', 'OY', 'UH', 'UW']
    consonant_list = ['P', 'B', 'CH', 'D', 'DH', 'F', 'G', 'HH', 'JH', 'K', 'L', 'M', 'N', 'NG', 'R', 'S',
                  'SH', 'T', 'TH', 'V', 'W', 'Y', 'Z', 'ZH']

    syllable_list = vowel_list + consonant_list

    words, pronounciations = [], []
    df = pd.DataFrame()
##    data = open(data)
    for line in data:
        w, p = line.split(':')
        p = p.rstrip('\n')
        words.append(w)
        pronounciations.append(p)

    for i in range(len(pronounciations)):
        syllables = pronounciations[i].split(' ')
        for j in range(len(syllables)):
            if syllables[j] in vowel_list:
                syllables[j] += '0'
        pronounciations[i] = ' '.join(syllables)
        
    df['w'] = words
    df['p'] = pronounciations

    for v in vowel_list:
        for i in range(1, 5):
            df[v+str(i)] = df['p'].apply(lambda p: has_v_as_ith_vowel_syllable(p, v, i))
        
    for s in syllable_list:
        for i in range(1, 5):
            for j in range(1, 3):
                name = str(s) + str(j) + 'before' + str(i) + 'vowel syllable'
                df[name] = df['p'].apply(lambda p: n_syllable_before_ith_vowel_syllable(p, s, i, j))

    for s in syllable_list:
        for i in range(1, 5):
            for j in range(1, 3):
                name = str(s) + str(j) + 'after' + str(i) + 'vowel syllable'
                df[name] = df['p'].apply(lambda p: n_syllable_after_ith_vowel_syllable(p, s, i, j))
                
    df['2 vowel syllables'] = df['p'].apply(lambda p: len([i for i in p.split(' ') if is_vowel(i)]) == 2)
    df['3 vowel syllables'] = df['p'].apply(lambda p: len([i for i in p.split(' ') if is_vowel(i)]) == 3)
    df['4 vowel syllables'] = df['p'].apply(lambda p: len([i for i in p.split(' ') if is_vowel(i)]) == 4)
            
    prediction = clf.predict(df[df.columns[2:]])
    return list(prediction)

##train('asset/training_data.txt', 'clf.data')
##print(test('asset/tiny_test.txt', 'clf.data'))

    
