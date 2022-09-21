from shiny import *
from shiny.types import FileInfo
import pandas as pd
from GA_select import parameters,split_transform_data, data_process
import os


app_ui = ui.page_fluid(
    ui.layout_sidebar(
        ui.panel_sidebar(
            
            ui.input_file("file1", "Choose CSV File", accept=[".csv"], multiple=False),
            ui.output_text("txt"),
            ui.input_checkbox("header", "Header", True),
            ui.input_numeric("Generations", "Number of generations", value = 10, step =1),
            ui.input_numeric("Size", "Population size", value = 10, step =1),
            ui.input_numeric("Crossover", "Crossover probability", value = 0.1, step =0.1),
            ui.input_numeric("Max_features_number", "Max Features number", value = None, step=10),
            
        
            ui.input_select("Targets", "Target Variable", choices = [], selected = None, 
            multiple=False),

            ui.input_selectize("Drop", "Drop Features", choices = [], selected = None, 
            multiple=True),
            
            ui.input_action_button("run", "Run Feature selection"),
            ui.output_text_verbatim("done", placeholder=True),

            
        ),
        ui.panel_main(ui.output_ui("contents")),
    )
)



def server(input: Inputs, output: Outputs, session: Session):
    @output
    @render.ui

    def contents():
        if input.file1() is None:
            return "Please upload a csv file and select the parameters"
        f: list[FileInfo] = input.file1()
        df = pd.read_csv(f[0]["datapath"], header=0 if input.header() else None)
        
        ui.nav_spacer()

        dataset = df.copy()
        
        options_target= dataset.columns.unique().tolist()
 
        ui.update_select("Targets", label = "Target Variable", 
                        choices = options_target),

        ui.update_selectize("Drop", label = "Drop Features", 
                        choices = options_target),




        # The @reactive.event() causes the function to run only when input.btn is
        # invalidated.
        @reactive.Effect
        @reactive.event(input.run)
        def _():
            #req(input.update())
            ui.update_action_button("run")
            Target = input.Targets.get()
            Generations = input.Generations.get()
            Size = input.Size.get()
            Crossover = input.Crossover.get()
            Max_features_number = input.Max_features_number.get()

        
            outdir = os.getcwd()

            columns_to_drop =  input.Drop.get()
            X, y, drop = data_process(dataset = dataset, target = Target, drop = columns_to_drop)

            clf, X_train_trans, X_test_trans, y_test, y_train, accuracy_no_GA  = split_transform_data(X = X, y= y)


            hr_start_time = parameters (generations = Generations, population_size =  Size, 
            crossover_probability = Crossover, max_features = Max_features_number,
            outdir = outdir, clf = clf, X_train_trans = X_train_trans, 
            X_test_trans =X_test_trans, y_test =y_test, y_train = y_train,  accuracy_no_GA =  accuracy_no_GA, additional_columns =columns_to_drop)
            ui.notification_show("Feature selection is done", type="message", duration=None)

    
        return ui.HTML(df.head(10).to_html(classes="table table-striped"))

    
    @output
    @render.text
    def txt():
        if input.file1() is not None:
            return f'Only a part of the data is shown'



app = App(app_ui, server)
