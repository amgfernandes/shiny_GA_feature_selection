from select import select
from shiny import *
from shiny.types import FileInfo
import pandas as pd
from GA_select import GA
import os
import matplotlib.pyplot as plt

'''Page layout'''
app_ui = ui.page_fluid(
    ui.layout_sidebar(
        ui.panel_sidebar(
            
            ui.input_file("file1", "Choose CSV File", accept=[".csv"], multiple=False),
            ui.output_text("txt"),
            ui.input_checkbox("header", "Header", True),
            ui.input_numeric("Generations", "Number of generations", value = 4, step =1, max=100),
            ui.input_numeric("Size", "Population size", value = 4, step =1),
            ui.input_numeric("Crossover", "Crossover probability", value = 0.1, step =0.1, max=0.9),
            ui.input_select("Targets", "Target Variable", choices = [], selected = None, 
            multiple=False),
            ui.input_selectize("Drop", "Drop Features", choices = [], selected = None, 
            multiple=True),
            ui.input_action_button("run", "Run Feature selection"),
            ui.output_text_verbatim("Done", placeholder=True),
            ui.output_table("table"),     
        ),
        ui.panel_main(ui.output_ui("contents"),
                      ui.output_plot("plot_fitness"),
                      ui.output_text("features"),
                      ui.output_text("to_drop"))
                    )
)

'''Server'''
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


        # The @reactive.event() causes the function to run only when input.run is
        # invalidated.
        @reactive.Effect
        @reactive.event(input.run)
        def table():
            ui.update_action_button("run")
            ui.notification_show("Feature selection is running. Results will be plotted when run is finished!", type="message", duration=None)        
            
            '''Get inputs'''
            Target = input.Targets.get()
            Generations = input.Generations.get()
            Size = input.Size.get()
            Crossover = input.Crossover.get()          
            outdir = os.getcwd()
            columns_to_drop =  input.Drop.get()
            
            X, y, drop = GA.data_process(dataset = dataset, target = Target, drop = columns_to_drop)
            clf, X_train_trans, X_test_trans, y_test, y_train, accuracy_no_GA  = GA.split_transform_data(X = X, y= y)


            hr_start_time, plot, selected_features = GA.running(generations = Generations, 
                                                            population_size =  Size, 
                                                            crossover_probability = Crossover, 
                                                            max_features = None,
                                                            outdir = outdir, 
                                                            clf = clf, 
                                                            X_train_trans = X_train_trans,
                                                            X_test_trans =X_test_trans, 
                                                            y_test =y_test, 
                                                            y_train = y_train,  
                                                            accuracy_no_GA =  accuracy_no_GA, 
                                                            additional_columns =columns_to_drop)
            
            
            ui.notification_show("Feature selection is done", type="message", duration=None)

            @output
            @render.plot(alt="Fitness plot")
            def plot_fitness():
                    fig = plot
                    return (fig)
  
            @output
            @render.text
            def features():                    
                sel = (', '.join(selected_features))
                return f'Selected features: "{sel}".'
                
                
            @output            
            @render.text
            def to_drop():               
                dropping = (', '.join(drop))
                return f'Removed features by user: "{dropping}"'
        
        return ui.HTML(df.head(10).to_html(classes="table table-striped"))


    @output
    @render.text
    def txt():
        if input.file1() is not None:
            return f'Only a part of the data is shown'
        

app = App(app_ui, server)
