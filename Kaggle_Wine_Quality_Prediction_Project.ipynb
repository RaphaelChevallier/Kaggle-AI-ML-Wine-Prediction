TASK{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "name": "TASK KAGGLE CHALLENGE - Wine Quality Prediction - Project.ipynb",
      "version": "0.3.2",
      "provenance": [],
      "private_outputs": true,
      "collapsed_sections": [
        "bzBbQPjTqGd3",
        "xG35AZ1aqGd4",
        "RayCVmJgrdHz",
        "oR-3amnaqGeO",
        "jxlMr97iqGei",
        "1svSbe48rkAL",
        "MxQbphc5r2fF",
        "FtA5kul0qGew",
        "oFrtqs9vqGez",
        "T_fkmW6Lr5sc",
        "8jw68DIRqGfH"
      ],
      "toc_visible": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "accelerator": "GPU"
  },
  "cells": [
    {
      "metadata": {
        "id": "8YZcPYIAqGbC",
        "colab_type": "text"
      },
      "cell_type": "markdown",
      "source": [
        "# KAGGLE CHALLENGE - Wine Quality Prediction - DATASET ANALYSIS TASK by Raphael C - \n",
        "\n"
      ]
    },
    {
      "metadata": {
        "id": "PNXUfs7uVjJX",
        "colab_type": "text"
      },
      "cell_type": "markdown",
      "source": [
        "# TASK Apply Supervised Learning algorithms in Machine Learning to real life data (wine) fitting simple models\n",
        "## Study the Wine Quality Data Set from UCI Machine Learning repository\n",
        "\n",
        "the data is available as two csv (with ';' as separators) at: \n",
        "\n",
        "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/\n",
        "\n",
        "one is about red wine the other about white wine\n",
        "\n",
        "\n",
        "**We will use the red wine dataset **. It is available on Dropbox:\n",
        "https://www.dropbox.com/s/s2lv2wjyr616vub/winequality-red.csv"
      ]
    },
    {
      "metadata": {
        "id": "2Rpa7e44q9hG",
        "colab_type": "text"
      },
      "cell_type": "markdown",
      "source": [
        "#Step 0 :  Import Key Python Libraries"
      ]
    },
    {
      "metadata": {
        "id": "DoloRktKqGbF",
        "colab_type": "code",
        "colab": {}
      },
      "cell_type": "code",
      "source": [
        "!pip install missingno"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "metadata": {
        "id": "R8l03wUUqGbO",
        "colab_type": "code",
        "colab": {}
      },
      "cell_type": "code",
      "source": [
        "import pandas as pd\n",
        "import numpy as np\n",
        "import matplotlib.pyplot as plt\n",
        "import seaborn as sns\n",
        "import missingno as ms\n",
        "%matplotlib inline"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "metadata": {
        "id": "DInPcxyBqGbT",
        "colab_type": "text"
      },
      "cell_type": "markdown",
      "source": [
        "#Step 1 :  Import the dataset"
      ]
    },
    {
      "metadata": {
        "id": "xhQhLNWGVf98",
        "colab_type": "code",
        "colab": {}
      },
      "cell_type": "code",
      "source": [
        "# Reading the dataset file from Dropbox - File was uploaded there for easy access from my Colab notebook\n",
        "!wget 'https://www.dropbox.com/s/s2lv2wjyr616vub/winequality-red.csv'"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "metadata": {
        "id": "gmWCjQ-FSWGB",
        "colab_type": "code",
        "colab": {}
      },
      "cell_type": "code",
      "source": [
        "!ls -l"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "metadata": {
        "id": "NGWNIC97qGbU",
        "colab_type": "code",
        "colab": {}
      },
      "cell_type": "code",
      "source": [
        "# Pandas provides great data types to manipulate data. The datatypes include Series and DataFrames.\n",
        "# Great functions include ways to read various sources like read_csv,read_excel, read_html etc.\n",
        "# The data is read and stored in the form of DataFrames.\n",
        "# reading the dataset into a Panda dataframe object named: data\n",
        "\n",
        "data = pd.read_csv('winequality-red.csv', sep=';')"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "metadata": {
        "id": "YOmpKXScdr3-",
        "colab_type": "text"
      },
      "cell_type": "markdown",
      "source": [
        "#Step 2 :  Analyze the Data"
      ]
    },
    {
      "metadata": {
        "id": "OYuetodJQWb9",
        "colab_type": "code",
        "colab": {}
      },
      "cell_type": "code",
      "source": [
        "print(\"Now checking how the data looks like:\")"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "metadata": {
        "id": "VQArfkvzqGbX",
        "colab_type": "code",
        "colab": {}
      },
      "cell_type": "code",
      "source": [
        "data.head(3)"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "metadata": {
        "id": "QbeOptuc7TjN",
        "colab_type": "code",
        "colab": {}
      },
      "cell_type": "code",
      "source": [
        "data.isnull().any()"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "metadata": {
        "id": "p3Dvngd_Zqk3",
        "colab_type": "code",
        "colab": {}
      },
      "cell_type": "code",
      "source": [
        "print(\"We can observe that there is NO missing values. Let's continue.\")"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "metadata": {
        "id": "huRcHobj8KtI",
        "colab_type": "code",
        "colab": {}
      },
      "cell_type": "code",
      "source": [
        "data.info()"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "metadata": {
        "id": "936l7an9_Aq1",
        "colab_type": "code",
        "colab": {}
      },
      "cell_type": "code",
      "source": [
        "data['quality'].hist(bins = 10, color = 'darkred')"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "metadata": {
        "id": "J0xLIgka-QO1",
        "colab_type": "code",
        "colab": {}
      },
      "cell_type": "code",
      "source": [
        "data['quality'].value_counts()"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "metadata": {
        "id": "9zlK0Dkn_6Ci",
        "colab_type": "code",
        "colab": {}
      },
      "cell_type": "code",
      "source": [
        "print(\"Wine quality histogram shows that regular wines (quality=5 and 6) make up the bulk of all wines with only 18 excellent wines, (quality=8) \")"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "metadata": {
        "id": "CskfxkFKDc1e",
        "colab_type": "code",
        "colab": {}
      },
      "cell_type": "code",
      "source": [
        "# GET SOME STATS ABOUT THOSE WINES\n",
        "data.describe()"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "metadata": {
        "id": "vP5QD4WqFfnZ",
        "colab_type": "code",
        "colab": {}
      },
      "cell_type": "code",
      "source": [
        "# pd.plotting.scatter_matrix(data, alpha = 0.3, figsize = (40,40), diagonal = 'kde');"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "metadata": {
        "id": "Y7h5ThrHGeEK",
        "colab_type": "code",
        "colab": {}
      },
      "cell_type": "code",
      "source": [
        "#-----------------------------LOOKING AT CORRELATION BETWEEN FEATURES ---------------------------------\n",
        "plt.figure(figsize=(16, 10))\n",
        "sns.heatmap(data.corr(),annot=True, cmap='coolwarm')\n",
        "plt.title('data.corr()')\n"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "metadata": {
        "id": "RayCVmJgrdHz",
        "colab_type": "text"
      },
      "cell_type": "markdown",
      "source": [
        "#Step 3 :   Feature Engineering and Data Preparation\n",
        "Splitting Features and Targets \n"
      ]
    },
    {
      "metadata": {
        "id": "yleBqUeKqGeL",
        "colab_type": "code",
        "colab": {}
      },
      "cell_type": "code",
      "source": [
        "data.info()"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "metadata": {
        "id": "rOgr2B_nIBjw",
        "colab_type": "code",
        "colab": {}
      },
      "cell_type": "code",
      "source": [
        "#------ Let's create a dataframe for the features (exclude the Quality) -----------------\n",
        "data_targets = data['quality']\n",
        "print(\"data_targets.shape\",data_targets.shape)\n",
        "\n",
        "#------ Let's create a dataframe for the targets (just the quality) -----------------\n",
        "data_features = data.drop(['quality'], axis = 1)\n",
        "print(\"data_features.shape\",data_features.shape)"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "metadata": {
        "id": "UY84d6dsTPYQ",
        "colab_type": "text"
      },
      "cell_type": "markdown",
      "source": [
        "#Step 4 :  Creating from the data a Training and a Test set. . .\n",
        "\n",
        "using scikit-learn "
      ]
    },
    {
      "metadata": {
        "id": "3bH6QYxaqGek",
        "colab_type": "code",
        "colab": {}
      },
      "cell_type": "code",
      "source": [
        "from sklearn.model_selection import train_test_split"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "metadata": {
        "id": "E22tvUEmqGel",
        "colab_type": "code",
        "colab": {}
      },
      "cell_type": "code",
      "source": [
        "X_train, X_test, y_train, y_test = train_test_split(data_features, \n",
        "                                                    data_targets, test_size=0.20, \n",
        "                                                    random_state=101)"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "metadata": {
        "id": "5jf2Zl2NqGeo",
        "colab_type": "code",
        "colab": {}
      },
      "cell_type": "code",
      "source": [
        "print(\"Training set has {} records.\".format(len(y_train)))\n",
        "print(\"Testing set has {} records.\".format(len(y_test)))"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "metadata": {
        "id": "jxlMr97iqGei",
        "colab_type": "text"
      },
      "cell_type": "markdown",
      "source": [
        "# Step - 5 :  Model Selection - (One Model to test)"
      ]
    },
    {
      "metadata": {
        "id": "1svSbe48rkAL",
        "colab_type": "text"
      },
      "cell_type": "markdown",
      "source": [
        "## Building a supervised learning model using scikit-learn - Logistic Regression model - "
      ]
    },
    {
      "metadata": {
        "id": "KKzr-_boqGes",
        "colab_type": "code",
        "colab": {}
      },
      "cell_type": "code",
      "source": [
        "from sklearn.linear_model import LogisticRegression\n",
        "# Build the Model.\n",
        "logmodel_displayname = 'LogisticRegression()'\n",
        "logmodel = LogisticRegression()\n",
        "logmodel.fit(X_train,y_train)"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "metadata": {
        "id": "S8bdMGNbqGet",
        "colab_type": "code",
        "colab": {}
      },
      "cell_type": "code",
      "source": [
        "predict =  logmodel.predict(X_test)\n",
        "predict[:10]"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "metadata": {
        "id": "J5CkHvS-qGev",
        "colab_type": "code",
        "colab": {}
      },
      "cell_type": "code",
      "source": [
        "y_test[:10] #---- checking by comparing predicted results(above) vs. actual results"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "metadata": {
        "id": "Xzj92CXQ7KNC",
        "colab_type": "code",
        "colab": {}
      },
      "cell_type": "code",
      "source": [
        "logmodel.score(X_train,y_train)"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "metadata": {
        "id": "_YKIZMdEqGev",
        "colab_type": "text"
      },
      "cell_type": "markdown",
      "source": [
        "Let's move on to evaluate our model."
      ]
    },
    {
      "metadata": {
        "id": "MxQbphc5r2fF",
        "colab_type": "text"
      },
      "cell_type": "markdown",
      "source": [
        "# Step - 6 : Evaluation - (One Model to test)"
      ]
    },
    {
      "metadata": {
        "id": "PgdudGTB_hmB",
        "colab_type": "code",
        "colab": {}
      },
      "cell_type": "code",
      "source": [
        "from sklearn.metrics import fbeta_score, accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, classification_report"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "metadata": {
        "id": "-oVRcg3iYJsc",
        "colab_type": "text"
      },
      "cell_type": "markdown",
      "source": [
        "## Eval Metrics descriptions"
      ]
    },
    {
      "metadata": {
        "colab_type": "text",
        "id": "A-ozbC3rCd39"
      },
      "cell_type": "markdown",
      "source": [
        "#### fbeta_score\n",
        "The F-beta score is the weighted harmonic mean of precision and recall, reaching its optimal value at 1 and its worst value at 0.\n",
        "\n",
        "####accuracy_score\n",
        "In multilabel classification, this function computes subset accuracy: the set of labels predicted for a sample must exactly match the corresponding set of labels in y_true.\n",
        "\n",
        "#### Precision Score\n",
        "The precision is the ratio tp / (tp + fp) where tp is the number of true positives and fp the number of false positives. The precision is intuitively the ability of the classifier not to label as positive a sample that is negative.\n",
        "The best value is 1 and the worst value is 0.\n",
        "\n",
        "#### Recall score\n",
        "The recall is the ratio tp / (tp + fn) where tp is the number of true positives and fn the number of false negatives. The recall is intuitively the ability of the classifier to find all the positive samples. The best value is 1 and the worst value is 0.\n",
        "\n",
        "#### f1_score\n",
        "The F1 score can be interpreted as a weighted average of the precision and recall, where an F1 score reaches its best value at 1 and worst score at 0. The relative contribution of precision and recall to the F1 score are equal. The formula for the F1 score is:\n",
        "                F1 = 2 \\* (precision \\* recall) / (precision + recall)\n",
        "\n",
        "#### Confusion Matrix\n",
        "True positive | False positive VS. False negative | True negative"
      ]
    },
    {
      "metadata": {
        "colab_type": "code",
        "id": "v9mQhNdICd3-",
        "colab": {}
      },
      "cell_type": "code",
      "source": [
        "# print(\"For model \", logmodel_displayname, \":\")\n",
        "# print(\"fbeta_score     : \", fbeta_score(y_test, predict, average='micro', beta=0.5))\n",
        "# print(\"accuracy_score  : % of correctly classified samples out of \", len(X_test),\": \", accuracy_score(y_test, predict, normalize=True))\n",
        "# print(\"precision_score : \", precision_score(y_test,predict, average= 'micro'))\n",
        "# print(\"recall_score    : \", recall_score(y_test,predict, average= 'micro'))\n",
        "# print(\"f1_score        : \", f1_score(y_test,predict, average= 'micro'))\n",
        "# print(\"confusion_matrix:\\n\", confusion_matrix(y_test, predict))"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "metadata": {
        "id": "t9YzG2swqGfF",
        "colab_type": "text"
      },
      "cell_type": "markdown",
      "source": [
        "## Eval Summary with: Classification Report (get all the above metrics at one go):"
      ]
    },
    {
      "metadata": {
        "id": "WpCBr2d2qGfG",
        "colab_type": "code",
        "colab": {}
      },
      "cell_type": "code",
      "source": [
        "print(\"For model \", logmodel_displayname, \":\")\n",
        "print(\"classification_report:\\n\", classification_report(y_test,predict))"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "metadata": {
        "id": "-eNCDFs8popU",
        "colab_type": "code",
        "colab": {}
      },
      "cell_type": "code",
      "source": [
        "conclusion = \"=============================CONCLUSION =========================================\\n\\\n",
        "\\n\\\n",
        "Classification on these wine quality values (3,4,5,6,7,8) is tricky since there is not enough\\n\\\n",
        "relevant samples of bad (q=3,4) or very good wines (q=7,8). Most of the data provided\\n\\\n",
        "is concentrated around the average wines (q=5,6).\\n\\\n",
        "Statistical evaluation results clearly shows that now we need to\\n\\\n",
        "-spend more time on finetuning data\\n\\\n",
        "-also bucketing data in simpler categories, say 3: Poor wines, Decent wines, Good wines would help\\n\\\n",
        "\\n\\\n",
        "================================================================================\"\n",
        "\n",
        "\n",
        "print(\"{}\".format(conclusion))"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "metadata": {
        "id": "HHigMpujoAaX",
        "colab_type": "text"
      },
      "cell_type": "markdown",
      "source": [
        "# Step - 5A/6A:  Models Selection & Evaluation - MULTIPLE MODELS\n"
      ]
    },
    {
      "metadata": {
        "id": "ZI9N8o0yoNio",
        "colab_type": "code",
        "colab": {}
      },
      "cell_type": "code",
      "source": [
        "from sklearn import svm, tree, linear_model, neighbors\n",
        "from sklearn import naive_bayes, ensemble, discriminant_analysis, gaussian_process\n",
        "\n",
        "#\n",
        "#--------- Building a list of ML algorthms to fit, test and generate evaluation metrics from\n",
        "#\n",
        "Algols = [\n",
        "            #Linear models\n",
        "            linear_model.LogisticRegression(),\n",
        "            linear_model.LogisticRegressionCV(),\n",
        "\n",
        "            #SVM\n",
        "            svm.LinearSVC(),\n",
        "\n",
        "            #Trees\n",
        "            tree.DecisionTreeClassifier(),\n",
        "\n",
        "            #Gaussian Processes\n",
        "            gaussian_process.GaussianProcessClassifier(),\n",
        "\n",
        "            #Navies Bayes\n",
        "            naive_bayes.BernoulliNB(),\n",
        "            naive_bayes.GaussianNB(),\n",
        "\n",
        "            #Nearest Neighbor\n",
        "            neighbors.KNeighborsClassifier(),\n",
        "\n",
        "          ]"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "metadata": {
        "id": "V0t0iGUtoNez",
        "colab_type": "code",
        "colab": {}
      },
      "cell_type": "code",
      "source": [
        "from sklearn.metrics import fbeta_score, accuracy_score, confusion_matrix, precision_score, recall_score, f1_score\n",
        "\n",
        "Algols_columns = []\n",
        "Algols_compare = pd.DataFrame(columns = Algols_columns)\n",
        "\n",
        "#\n",
        "#RUN EACH MODEL ONE MODEL AT A TIME AND CALCULATE EVALUATION METRICS\n",
        "#BUILD A MATRIX OF EVAL METRICS VALUES FOR EACH MODEL FOR DISPLAY\n",
        "#\n",
        "row_index = 0\n",
        "for alg in Algols:\n",
        "  \n",
        "    \n",
        "  predicted = alg.fit(X_train, y_train).predict(X_test) # run each model (fit) and calculate \n",
        "  Algo_type_of_model = alg.__class__\n",
        "\n",
        "  Algo_name = alg.__class__.__name__\n",
        "  Algols_compare.loc[row_index,'Model Type']   = Algo_type_of_model\n",
        "  Algols_compare.loc[row_index,'Name']         = Algo_name\n",
        "  Algols_compare.loc[row_index, 'fbeta_score'] = fbeta_score(y_test, predicted, average='micro', beta=0.5)\n",
        "  Algols_compare.loc[row_index, 'Accuracy']    = accuracy_score(y_test, predicted)\n",
        "  Algols_compare.loc[row_index, 'Precision']   = precision_score(y_test, predicted, average='micro')\n",
        "  Algols_compare.loc[row_index, 'Recall']      = recall_score(y_test, predicted, average='micro')\n",
        "  Algols_compare.loc[row_index, 'F1']          = f1_score(y_test, predicted, average='micro')\n",
        "\n",
        "  row_index+=1"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "metadata": {
        "id": "VpNZGpmDdlGG",
        "colab_type": "code",
        "colab": {}
      },
      "cell_type": "code",
      "source": [
        "#DISPLAY THE MATRIX OF RESULTS:\n",
        "\n",
        "#sort by Accuracy     \n",
        "Algols_compare.sort_values(by = ['Accuracy'], ascending = False, inplace = True)    \n",
        "\n",
        "#display output\n",
        "Algols_compare"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "metadata": {
        "colab_type": "code",
        "id": "st7XbnaTuUYY",
        "colab": {}
      },
      "cell_type": "code",
      "source": [
        "conclusion = \"=============================CONCLUSION =========================================\\n\\\n",
        "\\n\\\n",
        "Running a bunch of models (without really finetuning much the data at all at least allows us\\n\\\n",
        "to compare those models and their applicability to this multi-classification problem\\n\\\n",
        "\\n\\\n",
        "From the matrix above We can say that Naive Bayes ans LinearSVC did not perform well at all\\n\\\n",
        "\\n\\\n",
        "The most interesting models for this type of analysis are\\n\\\n",
        "- LogisticRegression\\n\\\n",
        "- DecisionTreeClassifier\\n\\\n",
        "- GaussianProcessClassifier\\n\\\n",
        "\\n\\\n",
        "====================================================================================\"\n",
        "\n",
        "\n",
        "print(\"{}\".format(conclusion))"
      ],
      "execution_count": 0,
      "outputs": []
    }
  ]
}
