"""
This file contains functions to process a dataset into the 6 levels specified in the perfors
paper on the Poverty of the Stimulus.
"""
import os

directory = "Adam/"
people = ["*MOT", "*URS", "*RIC", "*COL", "*CHI"]
def read_and_return(directory):
    speakers = {}
    struct = {}
    append_next = False
    for file_path in os.listdir(directory):
        with open("Adam/" + file_path, "r") as f:
            speakers[file_path] = []
            struct[file_path] = []
            for line in f:
                split = line.split("  ")
                if append_next and split[0][:4] == "%mor":
                    content = split[0].split("\t")[-1]
                    struct[file_path].append(content.split(" "))
                    
                elif split[0][:4] in people[:-1]:
                    speakers[file_path].append(split)
                    append_next = True
                else:
                    append_next = False
    return speakers, struct

class corpus:
    # stores all sentence forms in data
    def __init__(self):
        
        self.sentence_forms = {}
        for i in range(6): # init six levels
            self.sentence_forms[i + 1] = []

    def sort_sentence_types(self, types):
        for t in types:
            freq = types[t]
            if freq >= 500:
                self.sentence_forms[1].append(t)
            if freq >= 300:
                self.sentence_forms[2].append(t)
            if freq >= 100:
                self.sentence_forms[3].append(t)
            if freq >= 50:
                self.sentence_forms[4].append(t)
            if freq >= 10:
                self.sentence_forms[5].append(t)
                
            self.sentence_forms[6].append(t)


if __name__ == "__main__":
    # testing purposes all of the below:
    speakers, struct = read_and_return(directory) # this function was used before perfors sent his data
    types = {}
    num = 0 # used to find max pattern frequency
    for fp in struct:
        for segments in struct[fp]:
            t = "S"
            # print(segments)
            # exit()
            for s in segments[:-1]:
                token = s.split("|")[0].split(":")[0]
                t += "->" + token
            if t in types:
                types[t] += 1
            else:
                types[t] = 1
            if types[t] > num:
                num = types[t]

    print(len(types))
    print(num)

    
    data = corpus()
    data.sort_sentence_types(types)
    print(data.sentence_forms[1])
            
