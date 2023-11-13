# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
 
'''
Since problem classes are going to be inherited, trying some experiments with
multiple inheritance.

Experiment code is from the following source:

(1)
Title: How does Python's super() work with multiple inheritance?
Author: Callisto (author of question), rbp (author of answer)
Editor: Mateen Ulhaq (editor of question), Neuron (editor of answer)
URL: https://stackoverflow.com/questions/3277367/how-does-pythons-super-work-with-multiple-inheritance
Date published: July 18, 2010
Date edited: April 17, 2022 (question edited), August 30, 2021 (answer edited)
Date accessed: March 27, 2023
'''
class First(object):
  def __init__(self):
    print("First(): entering")
    super(First, self).__init__()
    print("First(): exiting")

  def other(self):
      print("first other called")

class Second(object):
  def __init__(self):
    print("Second(): entering")
    super(Second, self).__init__()
    print("Second(): exiting")

  def other2(self):
      print("Another other")

class Third(First, Second):
  def __init__(self):
    print("Third(): entering")
    super(Third, self).__init__()
    print("Third(): exiting")

  def other(self):
      super().other()


def tst_inher():
    th = Third()
    th.other()

