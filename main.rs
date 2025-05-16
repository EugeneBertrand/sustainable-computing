use std::env;
use std::fs::{OpenOptions};
use std::io::{self, Write, BufRead, BufReader};

fn add_task(task: String) -> io::Result<()> {
    let mut file = OpenOptions::new().append(true).create(true).open("todo.txt")?;
    writeln!(file, "{}", task)?;
    Ok(())
}

fn list_tasks() -> io::Result<()> {
    let file = OpenOptions::new().read(true).open("todo.txt")?;
    let reader = BufReader::new(file);
    for (i, line) in reader.lines().enumerate() {
        println!("{}: {}", i + 1, line?);
    }
    Ok(())
}

fn main() -> io::Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        println!("Usage: todo add <task> | list");
        return Ok(());
    }

    match args[1].as_str() {
        "add" => {
            if args.len() < 3 {
                println!("Please provide a task to add.");
            } else {
                add_task(args[2..].join(" "))?;
                println!("Task added.");
            }
        }
        "list" => list_tasks()?,
        _ => println!("Unknown command."),
    }

    Ok(())
}
