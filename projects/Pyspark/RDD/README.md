# Introduction to Spark programming


```
# /!\ RUN THIS CELL /!\

import hashlib

class TestFailure(Exception):
    pass
class PrivateTestFailure(Exception):
    pass

class Test(object):
    passed = 0
    numTests = 0
    grade = 0
    failFast = False
    private = False

    @classmethod
    def setFailFast(cls):
        cls.failFast = True

    @classmethod
    def setPrivateMode(cls):
        cls.private = True

    @classmethod
    def assertTrue(cls, result, msg="", points=0):
        cls.numTests += 1
        cls.grade += points
        if result == True:
            cls.passed += 1
            print("1 test passed.")
        else:
            print("1 test failed. " + msg)
        if cls.failFast:
            if cls.private:
                raise PrivateTestFailure(msg)
            else:
                raise TestFailure(msg)

    @classmethod
    def assertEquals(cls, var, val, msg="", points=0):
        cls.assertTrue(var == val, msg, points)

    @classmethod
    def assertEqualsHashed(cls, var, hashed_val, msg="", points=0):
        cls.assertEquals(cls._hash(var), hashed_val, msg, points)

    @classmethod
    def printStats(cls):
        print("{0} / {1} test(s) passed.".format(cls.passed, cls.numTests))
        print("Current grade: {0}.".format(cls.grade))

    @classmethod
    def _hash(cls, x):
        return hashlib.sha1(str(x).encode('utf-8')).hexdigest()
      
# /!\ RUN THIS CELL /!\
```


<style scoped>
  .ansiout {
    display: block;
    unicode-bidi: embed;
    white-space: pre-wrap;
    word-wrap: break-word;
    word-break: break-all;
    font-family: "Source Code Pro", "Menlo", monospace;;
    font-size: 13px;
    color: #555;
    margin-left: 4px;
    line-height: 19px;
  }
</style>
<div class="ansiout"></div>


# Word Count

The "Hello World!" of distributed programming is the wordcount. Basically, you want to count easily number of different words contained in an unstructured text. You will write some code to perform this task on the [Complete Works of William Shakespeare](http://www.gutenberg.org/ebooks/100) retrieved from [Project Gutenberg](http://www.gutenberg.org/wiki/Main_Page).

[Spark's Python API reference](https://spark.apache.org/docs/latest/api/python/pyspark.html#pyspark.RDD) could provide some help

### ** Part 1: Creating a base RDD and pair RDDs **

#### In this part of the lab, we will explore creating a base RDD with `parallelize` and using pair RDDs to count words.

#### We'll start by generating a base RDD by using a Python list and the `sc.parallelize` method.  Then we'll print out the type of the base RDD.


```
words_list = ['we', 'few', 'we', 'happy', 'few', "we", "band", "of", "brothers"]
words_RDD = sc.parallelize(words_list, 4)

# Print the type of words_RDD
print(type(words_RDD))
```


<style scoped>
  .ansiout {
    display: block;
    unicode-bidi: embed;
    white-space: pre-wrap;
    word-wrap: break-word;
    word-break: break-all;
    font-family: "Source Code Pro", "Menlo", monospace;;
    font-size: 13px;
    color: #555;
    margin-left: 4px;
    line-height: 19px;
  }
</style>
<div class="ansiout">&lt;class &#39;pyspark.rdd.RDD&#39;&gt;
</div>


We want to capitalize each word contained in a RDD. For such transformation, we use a `map`, as we want to transform a RDD of **n** elements into another RDD of **n** using a function that gets and returns one single element.

Please implement `capitalize`function in the cell below.

*Point: 0.5 pts*


```
def capitalize(word):
    """Capitalize lowercase `words`.

    Args:
        word (str): A lowercase string.

    Returns:
        str: A string which first letter is uppercase.
    """
    return word.capitalize()

print(capitalize('we'))

Test.assertEquals(capitalize('we'), 'We', "Capitalize", 0.5)
```


<style scoped>
  .ansiout {
    display: block;
    unicode-bidi: embed;
    white-space: pre-wrap;
    word-wrap: break-word;
    word-break: break-all;
    font-family: "Source Code Pro", "Menlo", monospace;;
    font-size: 13px;
    color: #555;
    margin-left: 4px;
    line-height: 19px;
  }
</style>
<div class="ansiout">We
1 test passed.
</div>


Apply `capitalize` to the base RDD, using a [map()](http://spark.apache.org/docs/latest/api/python/pyspark.html#pyspark.RDD.map) transformation that applies the `capitalize()` function to each element. Then call the [collect()](http://spark.apache.org/docs/latest/api/python/pyspark.html#pyspark.RDD.collect) action to retrieve the values of the transformed RDD, and print them.

*Point: 0.5 pts*


```
capital_RDD = words_RDD.map(capitalize).collect()
local_result = [capitalize(word) for word in words_list]
print(local_result)

Test.assertEqualsHashed(local_result, 'bd73c54004cc9655159aceb703bc14fe93369fb1',
                        'incorrect value for local_data', 0.5)
```


<style scoped>
  .ansiout {
    display: block;
    unicode-bidi: embed;
    white-space: pre-wrap;
    word-wrap: break-word;
    word-break: break-all;
    font-family: "Source Code Pro", "Menlo", monospace;;
    font-size: 13px;
    color: #555;
    margin-left: 4px;
    line-height: 19px;
  }
</style>
<div class="ansiout">[&#39;We&#39;, &#39;Few&#39;, &#39;We&#39;, &#39;Happy&#39;, &#39;Few&#39;, &#39;We&#39;, &#39;Band&#39;, &#39;Of&#39;, &#39;Brothers&#39;]
1 test passed.
</div>


Do the same using a lambda function

*Point: 0.5 pts*


```
capital_lambda_RDD = words_RDD.map(lambda x: x.capitalize()).collect()
local_result = [ (lambda x: x.capitalize())(word) for word in words_list]
print(local_result)

Test.assertEqualsHashed(local_result, 'bd73c54004cc9655159aceb703bc14fe93369fb1',
                        'incorrect value for capital_lambda_RDD', 0.5)
```


<style scoped>
  .ansiout {
    display: block;
    unicode-bidi: embed;
    white-space: pre-wrap;
    word-wrap: break-word;
    word-break: break-all;
    font-family: "Source Code Pro", "Menlo", monospace;;
    font-size: 13px;
    color: #555;
    margin-left: 4px;
    line-height: 19px;
  }
</style>
<div class="ansiout">[&#39;We&#39;, &#39;Few&#39;, &#39;We&#39;, &#39;Happy&#39;, &#39;Few&#39;, &#39;We&#39;, &#39;Band&#39;, &#39;Of&#39;, &#39;Brothers&#39;]
1 test passed.
</div>


Now use `map()` and a `lambda` function to return the number of characters in each word, and `collect` this result directly into a variable.

*Point: 0.5 pts*


```
plural_lengths = (words_RDD.map(lambda x: len(x)).collect())
print(plural_lengths)

Test.assertEqualsHashed(plural_lengths, '0772853c8e180c1bed8cfe9bde35aae79b277381',
                  'incorrect values for plural_lengths', 0.5)
```


<style scoped>
  .ansiout {
    display: block;
    unicode-bidi: embed;
    white-space: pre-wrap;
    word-wrap: break-word;
    word-break: break-all;
    font-family: "Source Code Pro", "Menlo", monospace;;
    font-size: 13px;
    color: #555;
    margin-left: 4px;
    line-height: 19px;
  }
</style>
<div class="ansiout">[2, 3, 2, 5, 3, 2, 4, 2, 8]
1 test passed.
</div>


To program a wordcount, we will need `pair RDD` objects. A pair RDD is an RDD where each element is a pair tuple `(k, v)` where `k` is the key and `v` is the value. In this example, we will create a pair consisting of `('<word>', 1)` for each word element in the RDD.

Create the pair RDD using the `map()` transformation with a `lambda()` on `words_RDD`.

*Point: 0.5 pts*


```
words_pair_RDD = words_RDD.map(lambda s: (s, 1))
print(words_pair_RDD.collect())

Test.assertEqualsHashed(words_pair_RDD.collect(), 'fb67a530034e01395386569ef29bf5565b503ec6',
                        "incorrect value for wrods_pair_RDD", 0.5)
```


<style scoped>
  .ansiout {
    display: block;
    unicode-bidi: embed;
    white-space: pre-wrap;
    word-wrap: break-word;
    word-break: break-all;
    font-family: "Source Code Pro", "Menlo", monospace;;
    font-size: 13px;
    color: #555;
    margin-left: 4px;
    line-height: 19px;
  }
</style>
<div class="ansiout">[(&#39;we&#39;, 1), (&#39;few&#39;, 1), (&#39;we&#39;, 1), (&#39;happy&#39;, 1), (&#39;few&#39;, 1), (&#39;we&#39;, 1), (&#39;band&#39;, 1), (&#39;of&#39;, 1), (&#39;brothers&#39;, 1)]
1 test passed.
</div>


Now, let's count the number of times a particular word appears in the RDD. There are multiple ways to perform the counting, but some are much less efficient or scalable than others.

A naive approach would be to `collect()` all of the elements and count them in the driver program. While this approach could work for small datasets, it is not scalable as the result of `collect()` would have to fit in the driver's memory. When you should use `collect()` with care, always asking yourself what is the size of data you want to retrieve.

In order to program a scalable wordcount, you will need to use parallel operations.

#### `groupByKey()` approach

An approach you might first consider is based on using the [groupByKey()](http://spark.apache.org/docs/latest/api/python/pyspark.html#pyspark.RDD.groupByKey) transformation. This transformation groups all the elements of the RDD with the same key into a single list, stored in one of the partitions. 
 
Use `groupByKey()` on `words_pair_RDD`
 to generate a pair RDD of type `('word', list)`.
 
 *Point: 0.5 pts*


```
words_grouped = words_pair_RDD.groupByKey()

for key, value in words_grouped.collect():
    print('{0}: {1}'.format(key, list(value)))
    
Test.assertEqualsHashed(sorted(words_grouped.mapValues(lambda x: list(x)).collect()),
                  'fdaad77fd81ef2df23d98ff7fd438fa700ca1fcf',
                  'incorrect value for words_grouped', 0.5)
```


<style scoped>
  .ansiout {
    display: block;
    unicode-bidi: embed;
    white-space: pre-wrap;
    word-wrap: break-word;
    word-break: break-all;
    font-family: "Source Code Pro", "Menlo", monospace;;
    font-size: 13px;
    color: #555;
    margin-left: 4px;
    line-height: 19px;
  }
</style>
<div class="ansiout">of: [1]
few: [1, 1]
brothers: [1]
we: [1, 1, 1]
band: [1]
happy: [1]
1 test passed.
</div>


Using the `groupByKey()` transformation results in an `pairRDD` containing words as keys, and Python iterators as values. Python iterators are a class of objects on which we can iterate, i.e.

    a = some_iterator()
    for elem in a:
        # do stuff with elem

Python lists and dictionnaries are iterators for example.

Now sum the iterator using a `map()` transformation. The result should be a pair RDD consisting of (word, count) pairs.

Hint: there exists a `sum` function
Hint 2: you want to perform an operation only on the values of the pairRDD. Take a look at [mapValues()](http://spark.apache.org/docs/latest/api/python/pyspark.html#pyspark.RDD.mapValues).

*Point: 0.5 pts*


```
word_grouped_counts = words_grouped.mapValues(lambda x: sum(x))
print(word_grouped_counts.collect())

Test.assertEqualsHashed(sorted(word_grouped_counts.collect()),
                  'c20f05d36e98ae399b2cbe5b6cb9bf01b675455a',
                  'incorrect value for word_grouped_counts', 0.5)
```


<style scoped>
  .ansiout {
    display: block;
    unicode-bidi: embed;
    white-space: pre-wrap;
    word-wrap: break-word;
    word-break: break-all;
    font-family: "Source Code Pro", "Menlo", monospace;;
    font-size: 13px;
    color: #555;
    margin-left: 4px;
    line-height: 19px;
  }
</style>
<div class="ansiout">[(&#39;of&#39;, 1), (&#39;few&#39;, 2), (&#39;brothers&#39;, 1), (&#39;we&#39;, 3), (&#39;band&#39;, 1), (&#39;happy&#39;, 1)]
1 test passed.
</div>


There are two problems with using `groupByKey()`:
  + The operation requires a lot of data movement to move all the values into the appropriate partitions (remember the cost of network communications!).
  + The lists can be very large. Consider a word count of English Wikipedia: the lists for common words (e.g., the, a, etc.) would be huge and could exhaust the available memory of a worker.

A better approach is to start from the pair RDD and then use the [reduceByKey()](http://spark.apache.org/docs/latest/api/python/pyspark.html#pyspark.RDD.reduceByKey) transformation to create a new pair RDD. The `reduceByKey()` transformation gathers together pairs that have the same key and applies the function provided to two values at a time, iteratively reducing all of the values to a single value. `reduceByKey()` operates by applying the function first within each partition on a per-key basis and then across the partitions, allowing it to scale efficiently to large datasets.

Compute the word count using `reduceByKey`

*Point: 0.5 pts*


```
word_counts = words_pair_RDD.reduceByKey(lambda x, y: x + y)
print(word_counts.collect())

Test.assertEqualsHashed(sorted(word_counts.collect()), 'c20f05d36e98ae399b2cbe5b6cb9bf01b675455a',
                  'incorrect value for word_counts', 0.5)
```


<style scoped>
  .ansiout {
    display: block;
    unicode-bidi: embed;
    white-space: pre-wrap;
    word-wrap: break-word;
    word-break: break-all;
    font-family: "Source Code Pro", "Menlo", monospace;;
    font-size: 13px;
    color: #555;
    margin-left: 4px;
    line-height: 19px;
  }
</style>
<div class="ansiout">[(&#39;of&#39;, 1), (&#39;few&#39;, 2), (&#39;brothers&#39;, 1), (&#39;we&#39;, 3), (&#39;band&#39;, 1), (&#39;happy&#39;, 1)]
1 test passed.
</div>


You should be able to perform the word count by composing functions, resulting in a smaller code. Use the `map()` on word RDD to create a pair RDD, apply the `reduceByKey()` transformation, and `collect` in one statement.

*Point: 0.5 pts*


```
word_counts_collected = (words_RDD
                         .map(lambda s: (s, 1))
                         .reduceByKey(lambda x, y: x + y)
                         .collect())

print(word_counts_collected)

Test.assertEqualsHashed(sorted(word_counts_collected), 'c20f05d36e98ae399b2cbe5b6cb9bf01b675455a',
                  'incorrect value for word_counts_collected', 0.5)
```


<style scoped>
  .ansiout {
    display: block;
    unicode-bidi: embed;
    white-space: pre-wrap;
    word-wrap: break-word;
    word-break: break-all;
    font-family: "Source Code Pro", "Menlo", monospace;;
    font-size: 13px;
    color: #555;
    margin-left: 4px;
    line-height: 19px;
  }
</style>
<div class="ansiout">[(&#39;of&#39;, 1), (&#39;few&#39;, 2), (&#39;brothers&#39;, 1), (&#39;we&#39;, 3), (&#39;band&#39;, 1), (&#39;happy&#39;, 1)]
1 test passed.
</div>


Compute the number of unique words using one of the RDD you have already created.

*Point: 0.5 pts*


```
unique_words = words_RDD.distinct().count()
print(unique_words)

Test.assertEquals(unique_words, 6, 'incorrect count of unique_words', 0.5)
```


<style scoped>
  .ansiout {
    display: block;
    unicode-bidi: embed;
    white-space: pre-wrap;
    word-wrap: break-word;
    word-break: break-all;
    font-family: "Source Code Pro", "Menlo", monospace;;
    font-size: 13px;
    color: #555;
    margin-left: 4px;
    line-height: 19px;
  }
</style>
<div class="ansiout">6
1 test passed.
</div>


Use a `reduce()` action to sum the counts in `wordCounts` and then divide by the number of unique words to find the mean number of words per unique word in `word_counts`.  First `map()` RDD `word_counts`, which consists of (key, value) pairs, to an RDD of values.

*Point: 0.5 pts*


```
from operator import add
total_count = (word_counts
              .map(lambda x: x[1])
              .reduce(add))

average = total_count / float(unique_words)
print(total_count)
print(round(average, 2))

Test.assertEquals(round(average, 2), 1.5, 'incorrect value of average', 0.5)
```


<style scoped>
  .ansiout {
    display: block;
    unicode-bidi: embed;
    white-space: pre-wrap;
    word-wrap: break-word;
    word-break: break-all;
    font-family: "Source Code Pro", "Menlo", monospace;;
    font-size: 13px;
    color: #555;
    margin-left: 4px;
    line-height: 19px;
  }
</style>
<div class="ansiout">9
1.5
1 test passed.
</div>


## Part 2: Apply word count to a file

In this section we will finish developing our word count application.  We'll have to build the `word_count` function, deal with real world problems like capitalization and punctuation, load in our data source, and compute the word count on the new data.

First, define a function for word counting. You should reuse the techniques that have been covered in earlier parts of this lab.  This function should take in an RDD that is a list of words like `words_RDD` and return a pair RDD that has all of the words and their associated counts.

*Point: 0.5 pts*


```
def word_count(word_list_RDD):
    """Creates a pair RDD with word counts from an RDD of words.

    Args:
        wordListRDD (RDD of str): An RDD consisting of words.

    Returns:
        RDD of (str, int): An RDD consisting of (word, count) tuples.
    """
    res = word_list_RDD.map(lambda s: (s, 1)).reduceByKey(lambda x, y: x + y)
    return res

print(word_count(words_RDD).collect())

Test.assertEqualsHashed(sorted(word_count(words_RDD).collect()),
                      'c20f05d36e98ae399b2cbe5b6cb9bf01b675455a',
                      'incorrect definition for word_count function', 0.5)
```


<style scoped>
  .ansiout {
    display: block;
    unicode-bidi: embed;
    white-space: pre-wrap;
    word-wrap: break-word;
    word-break: break-all;
    font-family: "Source Code Pro", "Menlo", monospace;;
    font-size: 13px;
    color: #555;
    margin-left: 4px;
    line-height: 19px;
  }
</style>
<div class="ansiout">[(&#39;of&#39;, 1), (&#39;few&#39;, 2), (&#39;brothers&#39;, 1), (&#39;we&#39;, 3), (&#39;band&#39;, 1), (&#39;happy&#39;, 1)]
1 test passed.
</div>


Real world data is more complicated than the data we have been using in this lab. Some of the issues we have to address are:
  + Words should be counted independent of their capitialization (e.g., Spark and spark should be counted as the same word).
  + All punctuation should be removed.
  + Any leading or trailing spaces on a line should be removed.
 
Define the function `removePunctuation` that converts all text to lower case, removes any punctuation, and removes leading and trailing spaces.  Use the Python [re](https://docs.python.org/2/library/re.html) module to remove any text that is not a letter, number, or space. Reading `help(re.sub)` might be useful.

If you have never used regex (regular expressions) before, you can refer to [Regular-expressions.info](http://www.regular-expressions.info/python.html)

In order to test your regular expressions, you can use [Regex Tester](http://www.regexpal.com)

Regex can be a bit obscure at beginning, don't hesitate to search in [StackOverflow](http://stackoverflow.com) or to ask me for some help.

*Point: 1 pts*


```
import re
import string

# Hint: string.punctuation contains all the punctuation symbols

def remove_punctuation(text):
    """Removes punctuation, changes to lower case, and strips leading and trailing spaces.

    Note:
        Only spaces, letters, and numbers should be retained.  Other characters should should be
        eliminated (e.g. it's becomes its).  Leading and trailing spaces should be removed after
        punctuation is removed.

    Args:
        text (str): A string.

    Returns:
        str: The cleaned up string.
    """
    text = text.lower()
    
    # remove punctuations with regex
    res = re.sub("[!\"#$%&'()*+,-./:;<=>?@[\]^_`{|}~]"  , "", text)
    
    # remove both leading and trailing spaces with regex
    res = re.sub(r"^\s+|\s+$","", res)
    return res

print(remove_punctuation('Hello World!'))
print(remove_punctuation(' No under_score!'))

Test.assertEquals(remove_punctuation("  Remove punctuation: there ARE trailing spaces. "),
                  'remove punctuation there are trailing spaces',
                  'incorrect definition for remove_punctuation function', 1.0)
```


<style scoped>
  .ansiout {
    display: block;
    unicode-bidi: embed;
    white-space: pre-wrap;
    word-wrap: break-word;
    word-break: break-all;
    font-family: "Source Code Pro", "Menlo", monospace;;
    font-size: 13px;
    color: #555;
    margin-left: 4px;
    line-height: 19px;
  }
</style>
<div class="ansiout">hello world
no underscore
1 test passed.
</div>


For the next part of this lab, we will use the [Complete Works of William Shakespeare](http://www.gutenberg.org/ebooks/100) from [Project Gutenberg](http://www.gutenberg.org/wiki/Main_Page). To convert a text file into an RDD, we use the `SparkContext.textFile()` method. We also apply the recently defined `remove_punctuation()` function using a `map()` transformation to strip out the punctuation and change all text to lowercase.  Since the file is large we use `take(15)`, instead of `collect()` so that we only print 15 lines.

Take a look at [zipWithIndex()](http://spark.apache.org/docs/latest/api/python/pyspark.html#pyspark.RDD.zipWithIndex) and [take()](http://spark.apache.org/docs/latest/api/python/pyspark.html#pyspark.RDD.take) to understand the print statement


```
file_path = '/FileStore/tables/shakespeare.txt'

shakespeare_RDD = (sc.textFile(file_path, 8)
                     .map(remove_punctuation))

print('\n'.join(shakespeare_RDD
                .zipWithIndex()  # to (line, lineNum) pairRDD
                .map(lambda x: '{0}: {1}'.format(x[1], x[0]))  # to 'lineNum: line'
                .take(15)))
```


<style scoped>
  .ansiout {
    display: block;
    unicode-bidi: embed;
    white-space: pre-wrap;
    word-wrap: break-word;
    word-break: break-all;
    font-family: "Source Code Pro", "Menlo", monospace;;
    font-size: 13px;
    color: #555;
    margin-left: 4px;
    line-height: 19px;
  }
</style>
<div class="ansiout">0: 1609
1: 
2: the sonnets
3: 
4: by william shakespeare
5: 
6: 
7: 
8: 1
9: from fairest creatures we desire increase
10: that thereby beautys rose might never die
11: but as the riper should by time decease
12: his tender heir might bear his memory
13: but thou contracted to thine own bright eyes
14: feedst thy lights flame with selfsubstantial fuel
</div>


Before we can use the `word_count()` function, we have to address two issues with the format of the RDD:
  + #### The first issue is that  that we need to split each line by its spaces.
  + #### The second issue is we need to filter out empty lines.
 
Apply a transformation that will split each element of the RDD by its spaces. For each element of the RDD, you should apply Python's string [split()](https://docs.python.org/2/library/string.html#string.split) function. You might think that a [map()](http://spark.apache.org/docs/latest/api/python/pyspark.html#pyspark.RDD.map) transformation is the way to do this, but think about what the result of the `split()` function will be: there is a better option.

Hint: remember the problem we had with `GroupByKey()`

*Point: 1 pts*


```
shakespeare_words_RDD = shakespeare_RDD.flatMap(lambda x: x.split(" "))
shakespeare_word_count_elem = shakespeare_words_RDD.count()
print(shakespeare_words_RDD.top(5))
print(shakespeare_word_count_elem)

# This test allows for leading spaces to be removed either before or after
# punctuation is removed.
Test.assertTrue(shakespeare_word_count_elem == 927694 or shakespeare_word_count_elem == 928908,
                'incorrect value for shakespeare_word_count_elem', 0.5)

Test.assertEqualsHashed(shakespeare_words_RDD.top(5),
                  'db4723f3c3190e69712b4cdf3ca67f21e86b04fd',
                  'incorrect value for shakespeare_words_RDD', 0.5)
```


<style scoped>
  .ansiout {
    display: block;
    unicode-bidi: embed;
    white-space: pre-wrap;
    word-wrap: break-word;
    word-break: break-all;
    font-family: "Source Code Pro", "Menlo", monospace;;
    font-size: 13px;
    color: #555;
    margin-left: 4px;
    line-height: 19px;
  }
</style>
<div class="ansiout">[&#39;zwaggerd&#39;, &#39;zounds&#39;, &#39;zounds&#39;, &#39;zounds&#39;, &#39;zounds&#39;]
927694
1 test passed.
1 test passed.
</div>


The next step is to filter out the empty elements.  Remove all entries where the word is `''`.

*Point: 1 pts*


```
shakespeare_nonempty_words_RDD = shakespeare_words_RDD.filter(lambda x: x != '')
shakespeare_nonempty_word_elem_count = shakespeare_nonempty_words_RDD.count()
print(shakespeare_nonempty_word_elem_count)

Test.assertEquals(shakespeare_nonempty_word_elem_count, 882996, 
                  'incorrect value for shakespeare_nonempty_word_elem_count', 1.0)
```


<style scoped>
  .ansiout {
    display: block;
    unicode-bidi: embed;
    white-space: pre-wrap;
    word-wrap: break-word;
    word-break: break-all;
    font-family: "Source Code Pro", "Menlo", monospace;;
    font-size: 13px;
    color: #555;
    margin-left: 4px;
    line-height: 19px;
  }
</style>
<div class="ansiout">882996
1 test passed.
</div>


You now have an RDD that contains only words.  Next, apply the `word_count()` function to produce a list of word counts. We can view the top 15 words by using the `takeOrdered()` action. However, since the elements of the RDD are pairs, you will need a custom sort function that sorts using the value part of the pair.

Use the `wordCount()` function and `takeOrdered()` to obtain the fifteen most common words and their counts.

*Point: 1 pts*


```
top15_words = word_count(shakespeare_nonempty_words_RDD).takeOrdered(15,key= lambda x:-x[1])
print('\n'.join(map(lambda x: '{0}: {1}'.format(x[0], x[1]), top15_words)))

Test.assertEqualsHashed(top15_words,
                        '4d052b4b7f38b6033b8b8aee9780108b24443e56',
                        'incorrect value for top15WordsAndCounts', 1.0)
```


<style scoped>
  .ansiout {
    display: block;
    unicode-bidi: embed;
    white-space: pre-wrap;
    word-wrap: break-word;
    word-break: break-all;
    font-family: "Source Code Pro", "Menlo", monospace;;
    font-size: 13px;
    color: #555;
    margin-left: 4px;
    line-height: 19px;
  }
</style>
<div class="ansiout">the: 27361
and: 26028
i: 20681
to: 19150
of: 17463
a: 14593
you: 13615
my: 12481
in: 10956
that: 10890
is: 9134
not: 8497
with: 7771
me: 7769
it: 7678
1 test passed.
</div>


You will notice that many of the words are common English words. These are called stopwords. In practice, when we do natural language processing, we filter these stopwords as they do not contain a lot of information.


```
# Print your current grade
# Warning, only valid if each cell have been run only once!!!
Test.printStats()
```


<style scoped>
  .ansiout {
    display: block;
    unicode-bidi: embed;
    white-space: pre-wrap;
    word-wrap: break-word;
    word-break: break-all;
    font-family: "Source Code Pro", "Menlo", monospace;;
    font-size: 13px;
    color: #555;
    margin-left: 4px;
    line-height: 19px;
  }
</style>
<div class="ansiout">17 / 17 test(s) passed.
Current grade: 10.0.
</div>



```

```


<style scoped>
  .ansiout {
    display: block;
    unicode-bidi: embed;
    white-space: pre-wrap;
    word-wrap: break-word;
    word-break: break-all;
    font-family: "Source Code Pro", "Menlo", monospace;;
    font-size: 13px;
    color: #555;
    margin-left: 4px;
    line-height: 19px;
  }
</style>

