def hendersond (a: int) -> str:
    '''
    It's random my dudes.
    '''
    import random
    myCoolNumbers = []
    random.seed = a
    for i in range(20 * a):
        myCoolNumbers.append(random.random())
    myCoolStrings = []
    for number in myCoolNumbers:
        numero = str(number)
        for x in range(len(numero)):
            if numero[x] == '0':
                myCoolStrings += 't'
            elif numero[x] == '1':
                myCoolStrings += 'h'
            elif numero[x] == '2':
                myCoolStrings += 'd'
                if random.random() > 0.6:
                    myCoolStrings += ' '
            elif numero[x] == '3':
                myCoolStrings += 'r'
                if random.random() > 0.6:
                    myCoolStrings += ' '
            elif numero[x] == '4':
                myCoolStrings += 'a'
            elif numero[x] == '5':
                myCoolStrings += 'y'
                if random.random() > 0.8:
                    myCoolStrings += ' '
            elif numero[x] == '6':
                myCoolStrings += 'i'
            elif numero[x] == '7':
                myCoolStrings += 'o'
            elif numero[x] == '8':
                myCoolStrings += 'e'
                if random.random() > 0.5:
                    myCoolStrings += ' '
            elif numero[x] == '9':
                myCoolStrings += 's'
        myCoolStrings += ' '
        
    myCoolOutput = ''.join(myCoolStrings)
    return myCoolOutput