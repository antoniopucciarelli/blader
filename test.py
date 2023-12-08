import customtkinter

app = customtkinter.CTk()
app.geometry("600x500")
app.title("CTk example")

frame = customtkinter.CTkFrame(master=app,
                               width=200,
                               height=200,
                               corner_radius=10,
                               fg_color="red")
frame.place(x=20, y=20)


app.mainloop()
