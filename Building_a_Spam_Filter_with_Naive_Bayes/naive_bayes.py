import pandas as pd
import re
import os


class Naive_Bayes_SMS_Filter:

    def __init__(self, filename, training_set_per=0.8, test_set_per=0.2):

        self.dataset = self.read_data(filename)
        # default training/test percentages for given dataset
        self.training_set_per = training_set_per
        self.test_set_per = 1 - test_set_per
        # setup training and test sets
        self.training_set, self.test_set = self.split_test_training_sets()
        # got to clean our training set, make msgs lower and remove punctuations
        self.clean_training_set()
        # setup msg vocabulary
        self.vocabulary = self.create_vocabulary()
        # setup clean training set
        self.training_set_clean = self.create_final_training_set()
        # Setup constants
        self.p_spam = None
        self.p_ham = None
        self.n_spam = None
        self.n_ham = None
        self.n_vocabulary = None
        self.alpha = None
        self.p_ham_denominator = None
        self.p_spam_denominator = None
        self.spam_messages = None
        self.ham_messages = None
        self.calculate_constants()
        # setup parameters
        self.parameters_ham, self.parameters_spam = self.calculate_parameters()

    def read_data(self, filename):
        dir_name = os.path.dirname(os.path.abspath(__file__))
        spam_file = os.path.join(dir_name, filename)
        sms_spam = pd.read_csv(spam_file, sep='\t',
                               header=None, names=['Label', 'SMS'])
        return sms_spam

    def split_test_training_sets(self, training_set_per=0.8, test_set_per=0.2):
        """
        this method will partition the dataset provided in first parameter
        into two separate sets, one for training and another for test based on
        the provided percentages

        Args:
            data (pandas DataFrame)- input data set
            training_set_per (float default 0.8 (80%)) = training set percentage
            test_set_per (float default 0.2 (20%)) = test set percentage
        Returns:
            two separate dataframes divided based on training_set_per and test_setper
            in that order
        """
        self.training_set_per = training_set_per
        self.test_set_per = test_set_per
        # randomize dataset
        randomized_dataset = self.dataset.sample(frac=1, random_state=1)

        training_index = round(len(randomized_dataset) * training_set_per)
        self.training_set = randomized_dataset[:training_index].reset_index(
            drop=True)
        self.test_set = randomized_dataset[training_index:].reset_index(
            drop=True)
        return self.training_set, self.test_set

    def clean_training_set(self):
        """
        # Step 1 - Make words all lower case
        # Step 2 - Remove punctuations
        """
        self.training_set['SMS'] = self.training_set['SMS'].str.lower()
        self.training_set['SMS'] = self.training_set['SMS'].str.replace(
            '\W', ' ')
        return self.training_set

    def create_vocabulary(self):
        self.training_set['SMS'] = self.training_set['SMS'].str.split()

        vocabulary = []
        for sms in self.training_set['SMS']:
            for word in sms:
                vocabulary.append(word)

        vocabulary = list(set(vocabulary))
        return vocabulary

    def create_final_training_set(self):

        # lets start by creating an empty dictionary with each word in our vacabulary as
        # key and 0s for each message. After that we will go through all sms and
        # update the count for each word
        word_counts_per_sms = {unique_word: [
            0] * len(self.training_set['SMS']) for unique_word in self.vocabulary}
        for index, sms in enumerate(self.training_set['SMS']):
            for word in sms:
                word_counts_per_sms[word][index] += 1
        # lets create pandas data frame with the our dictionary

        word_counts = pd.DataFrame(word_counts_per_sms)

        # lets concatinate orginal sms dataframe with the newly constructed one
        training_set_clean = pd.concat(
            [self.training_set, word_counts], axis=1)
        return training_set_clean

    def calculate_constants(self):

        # Isolating spam and ham messages first
        self.spam_messages = self.training_set_clean[self.training_set_clean['Label'] == 'spam']

        self.ham_messages = self.training_set_clean[self.training_set_clean['Label'] == 'ham']

        # P(Spam) and P(Ham)
        self.p_spam = len(self.spam_messages
                          ) / len(self.training_set_clean)
        self.p_ham = len(self.ham_messages) / \
            len(self.training_set_clean)

        # N_Spam
        n_words_per_spam_message = self.spam_messages['SMS'].apply(
            len)
        self.n_spam = n_words_per_spam_message.sum()

        # N_Ham
        n_words_per_ham_message = self.ham_messages['SMS'].apply(
            len)
        self.n_ham = n_words_per_ham_message.sum()

        # N_Vocabulary
        self.n_vocabulary = len(self.vocabulary)

        # Laplace smoothing
        self.alpha = 1

        # calculate denominators which will be constant
        self.p_ham_denominator = self.n_ham + \
            (self.alpha * self.n_vocabulary)
        self.p_spam_denominator = self.n_spam + \
            (self.alpha * self.n_vocabulary)

    def calculate_parameters(self):
        self.parameters_spam = {
            unique_word: 0 for unique_word in self.vocabulary}
        self.parameters_ham = {
            unique_word: 0 for unique_word in self.vocabulary}

        # Calculate parameters
        for word in self.vocabulary:
            self.parameters_spam[word] = (self.spam_messages[word].sum(
            ) + self.alpha) / self.p_spam_denominator
            self.parameters_ham[word] = (self.ham_messages[word].sum(
            ) + self.alpha) / self.p_ham_denominator
        return self.parameters_ham, self.parameters_spam

    def classify(self, message):
        '''
        message: messages that will be classifyed as either spam or ham
        Args:
            message (str): Message to classify
        Returns
            St: "spam" if the message is spam, "ham" if the message is not spam
                "needs human intervention" otherwise
        '''
        # remove punctuations
        message = re.sub('\W', ' ', message)
        message = message.lower().split()

        p_spam_given_message = self.p_spam
        p_ham_given_message = self.p_ham

        for word in message:
            if word in self.parameters_spam:
                p_spam_given_message *= self.parameters_spam[word]

            if word in self.parameters_ham:
                p_ham_given_message *= self.parameters_ham[word]

        if p_ham_given_message == p_spam_given_message:
            ret_val = "needs human intervention"
        else:
            ret_val = "ham" if p_ham_given_message > p_spam_given_message else "spam"
        return ret_val

    def check_accuracy(self):
        correct = 0
        self.test_set['predicted'] = self.test_set['SMS'].apply(self.classify)
        total = self.test_set.shape[0]

        for row in self.test_set.iterrows():
            row = row[1]
            if row['Label'] == row['predicted']:
                correct += 1

        print('Correct:', correct)
        print('Incorrect:', total - correct)
        print('Accuracy: {0:.2f}%'.format(correct/total * 100))
        return correct, (total - correct), (correct/total * 100)


if __name__ == '__main__':

    # read file adn print some basic information
    nb_sms_filter = Naive_Bayes_SMS_Filter("SMSSpamCollection")

    # We have everything now. Les go ahead and test
    test_msgs = ['WINNER!! This is the secret code to unlock the money: C3421.',
                 "Sounds good, Tom, then see u there", "you won the prize"]

    for msg in test_msgs:
        ret_val = nb_sms_filter.classify(msg)
        print(ret_val)
    nb_sms_filter.check_accuracy()
