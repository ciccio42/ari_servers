def pluralize(parola):
    vocali = ['a', 'e', 'i', 'o', 'u']
    suffissi_irregolari = {
        'io': 'i',
        'co': 'chi',
        'go': 'ghi',
        'ca': 'che',
        'ga': 'ghe',
        'cia': 'ce',
        'gia': 'ge',
        'scia': 'sce',
        'sc': 'schi'
    }

    parola = parola.lower().strip()

    if parola in suffissi_irregolari:
        return suffissi_irregolari[parola]

    if parola.endswith(('ch', 'gh')):
        return parola[:-1] + 'i'

    if parola.endswith('o') and not parola.endswith('ao') and parola[-2:] not in vocali:
        return parola[:-1] + 'i'

    if parola.endswith(('a', 'à')) and parola[-2:] not in vocali:
        return parola[:-1] + 'e'

    if parola.endswith('e') and parola[-2:] not in vocali:
        return parola[:-1] + 'i'

    return parola + 'i' if parola[-1] in vocali else parola + 'e'
