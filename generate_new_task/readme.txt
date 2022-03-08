links:
basic explanation: https://simpletransformers.ai/docs/t5-specifics/
full explanation: https://towardsdatascience.com/asking-the-right-questions-training-a-t5-transformer-model-on-a-new-task-691ebba2d72c


1. train folder with all the songs you want to use for both train and val (can be changed later)
2. run test.py
3. run predict.py
4. the task name is called MOR (just to see if it work - should be changed to "genre matching" or whatever we choose)
5. the tagging is 1 for every song (to see if it tags new songs the same, should be changed to the genre name)
6. if it ask for an api or something:
    click on option 3, you should see the result in
    - outputs > generated_genres AFTER running predict.py OR
    - in the terminal AFTER running test.py - you will see the array of ones
        (the real resutls) and the array of the predicted resutls below it, 
        and you should see how many matches there are)
    - if it doesnt work i will search for my api that you need.

code explanation:
t5 should get a pandas dataframe with 3 columns:
 - prefix: the task
 - input text: the song
 - target text: the genre

 then we train on this and predict on a new data.

 ignore model args - we can change them later/add others.  