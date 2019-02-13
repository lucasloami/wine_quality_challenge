# HELPER FUNCTION DEFINITIONS

def plot_learning_curve(title, X, y, ylim=None, cv=None, n_jobs=1, 
                        train_sizes=np.linspace(.1, 1.0, 5), max_depths=None, subset=False):
  fig = plt.figure(figsize=(12,7))

  if max_depths is None:
    max_depths = [5,10,20,30]

  if subset:
    merged_df = pd.concat((X,y), axis=1, join='inner')
    subset_df = merged_df.sample(n=4000, random_state=20)
    X = subset_df[X.columns]
    y = subset_df[y.name]

  for k, depth in enumerate(max_depths):

    # Create a Decision tree regressor at max_depth = depth
    estimator = tree.DecisionTreeClassifier(max_depth = depth, criterion="entropy")
    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y,
                                                          cv=cv, n_jobs=n_jobs, 
                                                          train_sizes=train_sizes,
                                                           scoring='accuracy')
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    ax = fig.add_subplot(2,2,k+1)

    ax.fill_between(train_sizes, train_scores_mean - train_scores_std,
                   train_scores_mean + train_scores_std, alpha=0.1,
                   color="r")
    ax.fill_between(train_sizes, test_scores_mean - test_scores_std,
                   test_scores_mean + test_scores_std, alpha=0.1, color="g")
    ax.plot(train_sizes, train_scores_mean, 'o-', color="r",
           label="Training score")
    ax.plot(train_sizes, test_scores_mean, 'o-', color="g",
           label="Cross-validation score")

    # Labels
    ax.set_title('max_depth = %s'%(depth))
    ax.set_xlabel('Number of Training Points')
    ax.set_ylabel('Score')
    ax.set_xlim([0, X.shape[0]*0.8])
    ax.set_ylim([0.5, 1.05])

  # Visual aesthetics
  ax.legend(bbox_to_anchor=(1.05, 1.05), loc='lower left', borderaxespad = 0.)

  fig.suptitle(title, fontsize = 16, y = 1.03)

  plt.show()

def plot_model_complexity(X, y):
  """ Calculates the performance of the model as model complexity increases.
      The learning and testing errors rates are then plotted. """

  # Vary the max_depth parameter from 1 to 10
  max_depth = np.arange(1,20)
  clf = tree.DecisionTreeClassifier(criterion="entropy")

  # Calculate the training and testing scores
  train_scores, test_scores = validation_curve(clf, X, y, 
                                                      param_name = "max_depth", 
                                                      param_range = max_depth, 
                                                      cv = 10, scoring = 'accuracy')

  # Find the mean and standard deviation for smoothing
  train_mean = np.mean(train_scores, axis=1)
  train_std = np.std(train_scores, axis=1)
  test_mean = np.mean(test_scores, axis=1)
  test_std = np.std(test_scores, axis=1)

  # Plot the validation curve
  plt.figure(figsize=(10, 5))
  plt.title('Decision Tree Classifier Complexity Performance')
  plt.plot(max_depth, train_mean, 'o-', color = 'r', label = 'Training Score')
  plt.plot(max_depth, test_mean, 'o-', color = 'g', label = 'Validation Score')
  plt.fill_between(max_depth, train_mean - train_std, \
      train_mean + train_std, alpha = 0.15, color = 'r')
  plt.fill_between(max_depth, test_mean - test_std, \
      test_mean + test_std, alpha = 0.15, color = 'g')

  # Visual aesthetics
  plt.legend(loc = 'lower right')
  plt.xlabel('Maximum Depth')
  plt.ylabel('Score')
  plt.ylim([0.87,0.99])
  plt.show()

def plot_roc_curve(y_true, y_pred, n_classes=1):
  # Determine the false positive and true positive rates
  fpr, tpr, _ = roc_curve(y_test, y_pred)

  # Calculate the AUC
  roc_auc = auc(fpr, tpr)
  print 'ROC AUC: %0.2f' % roc_auc

  plt.figure()
  plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
  plt.plot([0, 1], [0, 1], 'k--')
  plt.xlim([0.0, 1.0])
  plt.ylim([0.0, 1.05])
  plt.xlabel('False Positive Rate')
  plt.ylabel('True Positive Rate')
  plt.title('ROC Curve')
  plt.legend(loc="lower right")
  plt.show()
