# Workshop: Dashboarding


## Getting Started

1. Install the requirements

    ```bash
    pip install -r requirements.txt
    ```

1. Start the application

    ```bash
    streamlit run src/app.py
    ```

1. The application should have opened in you web browser. If not, open one and click on [here](http://localhost:8501)


## Usage of the Dashboard

1. Select the date of the optimization.
2. Select the two control algorithms to compare.
3. Inspect the behaviour on the chart.
4. If you need to debug anything, turning on the Debug checkbox will show some extra information.
5. Now you are ready to play around with the extra parameters.


## Points of improvements

There are several points where this dashboard could be improved and extended. We list just some of
them here, but feel free to build upon this dashboard for your case.

- The metrics bar does not show any control related metrics. Try to add some that help decide which algorithm works better.
- The costs are calculated based on the forecasted demand and generation. Extend the metrics with the realized savings and check how far they are from the forecasted ones.
- The forecasts for next day are very simple naive seasonal models. Can you improve the controls with better forecast?
- The control algorithms are very simple and the random controller is not very realistic. Try to create a better controller and evaluate their improvement using the added metrics.
- Even though day-night pricing is still common in Europe, real time pricing is a possibility in many countries. Try to extend the tariff choices to evaluate your controller in different environments.
- The simulation implements an ideal storage without any losses. Can you make the simulation more realistic?
