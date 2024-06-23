def add_tags(file, tgtlang):
    l = []
    with open(file, 'r') as f:
        l = f.readlines()
        l[0] = l[0][:-1] + '\tsrc_lang\ttgt_lang\n'
        for i in range (1, len(l)):
            l[i] = l[i][:-1]+'\ten\t' + tgtlang + '\n'

    with open(file, 'w') as f:
        f.writelines(l)
    print("---Added lang tags in", file)

add_tags("${MUSTC_ROOT}/en-de/train_st.tsv", "de")
add_tags("${MUSTC_ROOT}/en-de/dev_st.tsv", "de")
add_tags("${MUSTC_ROOT}/en-de/tst-COMMON_st.tsv", "de")
add_tags("${MUSTC_ROOT}/en-fr/train_st.tsv", "fr")
add_tags("${MUSTC_ROOT}/en-fr/dev_st.tsv", "fr")
add_tags("${MUSTC_ROOT}/en-fr/tst-COMMON_st.tsv", "fr")
add_tags("${MUSTC_ROOT}/en-ru/train_st.tsv", "ru")
add_tags("${MUSTC_ROOT}/en-ru/dev_st.tsv", "ru")
add_tags("${MUSTC_ROOT}/en-ru/tst-COMMON_st.tsv", "ru")
add_tags("${MUSTC_ROOT}/en-nl/train_st.tsv", "nl")
add_tags("${MUSTC_ROOT}/en-nl/dev_st.tsv", "nl")
add_tags("${MUSTC_ROOT}/en-nl/tst-COMMON_st.tsv", "nl")

