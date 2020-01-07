import random

def read_csv_file(fileName):
    ret = []
    with open(fileName, "r") as fd:
        for line in fd:
            ret.append(line.strip().split(","))
    return ret

def main(perturbation=0.0):
    tests = read_csv_file("data/mnist_test.csv")
    upper_array = "static float mnist_test[%d][%d] = {\n" % (len(tests), len(tests[0]))
    correct_class = "static int correct_class[%d] = {\n" % (len(tests))
    random_pixels = "static int random_pixels[%d][10] = {\n" % (len(tests))
    for test in tests:
        upper_array += "{" + ",".join(test[1:]) + "},\n"
        correct_class += test[0] + ",\n"
        random_pixels += "{" + ",".join(map(str, random.sample(range(784), 10))) + "},\n"
    upper_array = upper_array.rstrip(",") + "};\n"
    correct_class = correct_class.rstrip(",") + "};"
    random_pixels = random_pixels.rstrip(",") + "};"
    print(upper_array)
    print(correct_class)
    print(random_pixels)

if __name__ == "__main__":
    main()
