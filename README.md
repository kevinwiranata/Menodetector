# [Menopredictor] Menopause Symptom Predictor

Predict the severity of Menopausal symptoms based on a patient's lifestyle!

<h1>Setup</h1>

<p>Menopredictor requires Python 3.9 or higher. To check your version of Python, run either:</p>

<div><pre><code>&gt;&gt;&gt; python --version
&gt;&gt;&gt; python3 --version
</code></pre></div>

<p>We recommend creating a global Menopause workspace directory that you will use for all modules.</p>

<div><pre><code>&gt;&gt;&gt; mkdir workspace; cd workspace
</code></pre></div>

<p>You can clone your own version by running the following command:</p>

<div><pre><code>&gt;&gt;&gt; git clone https://github.com/kevinwiranata/Menodetector.git
&gt;&gt;&gt; cd Menopredictor
</code></pre></div>

<p>We also highly recommend setting up a <em>virtual environment</em>. The virtual environment lets you install packages that are only used for your assignments and do not impact the rest of the system. We suggest venv or anaconda.</p>

<p>For example, if you choose venv, run the following command:</p>

<div><pre><code>&gt;&gt;&gt; python -m venv venv
&gt;&gt;&gt; source venv/bin/activate
</code></pre></div>

<p>The first line should be run only once, whereas the second needs to be run whenever you open a new terminal to get started for the class. You can tell if the second line works by checking if your terminal starts with <code>(venv)</code>. See <a href="https://docs.python.org/3/library/venv.html">https://docs.python.org/3/library/venv.html</a> for further instructions on how this works.</p>

<p>The last step is to install packages. There are several packages used and you can install them in your virtual environment by running:</p>

<div><pre><code>&gt;&gt;&gt; python -m pip install -r requirements.txt
</code></pre></div>

**Alternatively**, if using VSCode, you can create a virtual environment using the Command Palette, by carrying out the following actions:

<div><pre><code>Cmd+Shift+P --> Python: Create Environment --> venv --> python --> select requirements.txt
</code></pre></div>

<h2>Formatting</h2>

- We use [Python Black](https://pypi.org/project/black/) as our formatting editor, which can installed as as [VSCode Extension](https://marketplace.visualstudio.com/items?itemName=ms-python.black-formatter).
- We recommend enabling `Format on Save` and `Format on Type` in VSCode as well


<h2> Running</h2>
To run the code simply run the <code>run_model.py</code> file.
<div><pre><code>python3 run_model.py
</code></pre></div>

There are several arg options when running this file:
- `--grid_search` to run grid search
- `--use_optimal_params` to use optimal params
- `--epochs`, `--learning_rate`, `--hidden_layer_size`, `--batch_size` to specificy hyperparameters

You can also run <code> run_model.py -h </code> to see all the available arguments

Example commands:
1. Run grid search
<div><pre><code>python3 run_model.py -g True
</code></pre></div>

2. Use optimal parameters (no grid search)
<div><pre><code>python3 run_model.py -op True
</code></pre></div>

3. Specificy Hyperparameters
<div><pre><code>python3 run_model.py -e 10 -lr 0.001 -hl 100 -b 32
</code></pre></div>