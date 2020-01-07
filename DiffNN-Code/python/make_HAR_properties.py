def read_csv_file(fileName):
    ret = []
    with open(fileName, "r", encoding="utf-8") as fd:
        for line in fd:
            ret.append(list(filter(lambda x: bool(x), line.strip().split(" "))))
    return ret

def mainForDeepPoly():
    num_tests = 100
    tests = read_csv_file("data/UCI_HAR_Dataset/test/X_test.txt")
    classes = read_csv_file("data/UCI_HAR_Dataset/test/y_test.txt")
    for i in range(num_tests):
        print(str(int(classes[i][0])-1) + "," + ",".join(tests[i]))

def main():
    num_tests = 100
    tests = read_csv_file("data/UCI_HAR_Dataset/test/X_test.txt")
    classes = read_csv_file("data/UCI_HAR_Dataset/test/y_test.txt")
    upper_array = "static float HAR_test[%d][%d] = {\n" % (num_tests, len(tests[0]))
    correct_class = "static int HAR_correct_class[%d] = {\n" % (num_tests)
    for i in range(num_tests):
        upper_array += "{" + ",".join(tests[i]) + "},\n"
        correct_class += str(int(classes[i][0]) - 1) + ",\n"
    upper_array = upper_array.rstrip(",") + "};\n"
    correct_class = correct_class.rstrip(",") + "};"
    print(upper_array)
    print(correct_class)

if __name__ == "__main__":
    mainForDeepPoly()
